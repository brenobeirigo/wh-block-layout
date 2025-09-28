import os
import gurobipy as gp
from gurobipy import GRB
import math
import itertools as it
from typing import Dict, List, Optional, Tuple, Literal, Any, Union
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SolverConfig:
    """Configuration parameters for the solver."""
    mip_gap: float = 0.02
    time_limit: Optional[int] = None
    log_to_console: bool = True
    write_lp: Optional[str] = None
    
    def apply_to_model(self, model: gp.Model) -> None:
        """Apply configuration to Gurobi model."""
        model.Params.OutputFlag = 1 if self.log_to_console else 0
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = self.mip_gap


@dataclass
class GeometryConfig:
    """Configuration for geometric constraints and bounds."""
    building_x: float
    building_y: float
    s_min: float = 0.7
    s_max: float = 1.6
    use_perimeter: bool = True
    rho_perimeter: float = 1.30
    use_aspect: bool = True
    aspect_ratio_limit: float = 3.0
    default_clearance: float = 0.0
    area_calculation: Literal["exact", "envelope"] = "envelope"
    area_band_pct: Optional[float] = None


@dataclass
class ProblemData:
    """Core problem data."""
    dept_list: List[str]
    areas_m2: Dict[str, float]
    weights: pd.DataFrame
    pair_clearances: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None
    anchors: Optional[Dict[str, Dict[str, float]]] = None
    design_skeleton: Optional[List[Tuple[str, str, str]]] = None
    fixed_departments: Optional[Dict[str, Dict[str, float]]] = None


class OptimizationSolver(ABC):
    """Abstract base class for optimization solvers."""
    
    def __init__(self, problem_data: ProblemData, solver_config: SolverConfig):
        self.problem_data = problem_data
        self.solver_config = solver_config
        self.model = None
        self.variables = {}
        self.is_built = False
        self.is_solved = False
        
    @abstractmethod
    def build_model(self) -> None:
        """Build the optimization model."""
        pass
    
    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """Solve the model and return results."""
        pass
    
    def _create_base_model(self, name: str) -> gp.Model:
        """Create base Gurobi model with configuration."""
        model = gp.Model(name)
        self.solver_config.apply_to_model(model)
        return model
    
    def _write_model(self) -> None:
        """Write model to file if specified."""
        if self.solver_config.write_lp and self.model:
            dir_path = os.path.dirname(self.solver_config.write_lp)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            self.model.write(self.solver_config.write_lp)
    
    def get_solution_dict(self) -> Dict[str, Any]:
        """Extract solution values from variables."""
        if not self.is_solved:
            raise RuntimeError("Model has not been solved yet")
            
        solution = {}
        for var_name, var_dict in self.variables.items():
            if hasattr(var_dict, 'keys'):  # Multi-dimensional variable
                solution[var_name] = {key: var.X for key, var in var_dict.items()}
            else:  # Single variable
                solution[var_name] = var_dict.X
        return solution


class MILPSolver(OptimizationSolver):
    """Mixed Integer Linear Programming solver."""
    
    def __init__(self, problem_data: ProblemData, geometry_config: GeometryConfig, 
                 solver_config: SolverConfig):
        super().__init__(problem_data, solver_config)
        self.geometry_config = geometry_config
    
    def build_model(self) -> None:
        """Build MILP model for block layout problem."""
        if self.is_built:
            return
            
        # Validate inputs
        self._validate_inputs()
        
        # Create model
        self.model = self._create_base_model("block_layout_milp")
        
        # Build components
        self._create_variables()
        self._add_geometry_constraints()
        self._add_distance_constraints()
        self._add_separation_constraints()
        self._apply_design_skeleton()
        self._apply_anchors()
        self._apply_fixed_departments()
        self._set_objective()
        
        # Write model if requested
        self._write_model()
        
        self.is_built = True
    
    def solve(self) -> Dict[str, Any]:
        """Solve the MILP model."""
        if not self.is_built:
            self.build_model()
            
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            self.is_solved = True
            return {
                'status': 'optimal',
                'objective_value': self.model.ObjVal,
                'variables': self.get_solution_dict(),
                'solve_time': self.model.Runtime
            }
        elif self.model.Status == GRB.TIME_LIMIT:
            return {
                'status': 'time_limit',
                'objective_value': self.model.ObjVal if self.model.SolCount > 0 else None,
                'variables': self.get_solution_dict() if self.model.SolCount > 0 else None,
                'solve_time': self.model.Runtime
            }
        else:
            return {
                'status': 'infeasible_or_error',
                'gurobi_status': self.model.Status,
                'solve_time': self.model.Runtime
            }
    
    def _validate_inputs(self) -> None:
        """Validate problem inputs."""
        dept_set = set(self.problem_data.dept_list)
        weight_rows = set(self.problem_data.weights.index)
        weight_cols = set(self.problem_data.weights.columns)
        
        assert dept_set.issubset(weight_rows), "weights missing departments in rows"
        assert dept_set.issubset(weight_cols), "weights missing departments in columns"
    
    def _create_variables(self) -> None:
        """Create decision variables."""
        I = self.problem_data.dept_list
        unordered_pairs = [(i, j) for i, j in it.combinations(I, 2)]
        
        # Rectangle coordinates
        self.variables['x_west'] = self.model.addVars(I, lb=0.0, ub=self.geometry_config.building_x, name="x_west")
        self.variables['x_east'] = self.model.addVars(I, lb=0.0, ub=self.geometry_config.building_x, name="x_east")
        self.variables['y_south'] = self.model.addVars(I, lb=0.0, ub=self.geometry_config.building_y, name="y_south")
        self.variables['y_north'] = self.model.addVars(I, lb=0.0, ub=self.geometry_config.building_y, name="y_north")
        
        # Dimensions and centers
        self.variables['width'] = self.model.addVars(I, lb=0.0, name="width")
        self.variables['height'] = self.model.addVars(I, lb=0.0, name="height")
        self.variables['center_x'] = self.model.addVars(I, lb=0.0, ub=self.geometry_config.building_x, name="center_x")
        self.variables['center_y'] = self.model.addVars(I, lb=0.0, ub=self.geometry_config.building_y, name="center_y")
        
        # Distance variables
        self.variables['dist_x'] = self.model.addVars(unordered_pairs, lb=0.0, name="dist_x")
        self.variables['dist_y'] = self.model.addVars(unordered_pairs, lb=0.0, name="dist_y")
        
        # Binary indicators
        self.variables['is_east_of'] = self.model.addVars(I, I, vtype=GRB.BINARY, name="is_east_of")
        self.variables['is_north_of'] = self.model.addVars(I, I, vtype=GRB.BINARY, name="is_north_of")
    
    def _add_geometry_constraints(self) -> None:
        """Add geometric constraints."""
        I = self.problem_data.dept_list
        areas = self.problem_data.areas_m2
        
        # Precompute geometric parameters
        sqrt_area = {i: math.sqrt(areas[i]) for i in I}
        perim_min = {i: 4.0 * sqrt_area[i] for i in I}
        perim_max = {i: self.geometry_config.rho_perimeter * perim_min[i] for i in I}
        
        # Determine which departments have fixed dimensions
        fixed_set = set(self.problem_data.fixed_departments.keys()) if self.problem_data.fixed_departments else set()
        skip_surrogates = self._get_skip_surrogates_dict(fixed_set)
        
        for i in I:
            # Define dimensions from coordinates
            self.model.addConstr(self.variables['width'][i] == self.variables['x_east'][i] - self.variables['x_west'][i], name=f"def_width[{i}]")
            self.model.addConstr(self.variables['height'][i] == self.variables['y_north'][i] - self.variables['y_south'][i], name=f"def_height[{i}]")
            
            # Define centers
            self.model.addConstr(self.variables['center_x'][i] == 0.5 * (self.variables['x_west'][i] + self.variables['x_east'][i]), name=f"def_center_x[{i}]")
            self.model.addConstr(self.variables['center_y'][i] == 0.5 * (self.variables['y_south'][i] + self.variables['y_north'][i]), name=f"def_center_y[{i}]")
            
            # Ordering constraints
            self.model.addConstr(self.variables['x_west'][i] <= self.variables['x_east'][i], name=f"x_order[{i}]")
            self.model.addConstr(self.variables['y_south'][i] <= self.variables['y_north'][i], name=f"y_order[{i}]")
            
            # Area and shape constraints (if not skipped)
            if not skip_surrogates[i]:
                self._add_shape_constraints(i, sqrt_area[i], perim_min[i], perim_max[i])
    
    def _add_shape_constraints(self, dept: str, sqrt_area: float, perim_min: float, perim_max: float) -> None:
        """Add shape constraints for a department."""
        if self.geometry_config.area_calculation == "envelope":
            # Linear bounds around sqrt(area)
            self.model.addConstr(self.variables['width'][dept] >= self.geometry_config.s_min * sqrt_area, name=f"width_lb[{dept}]")
            self.model.addConstr(self.variables['width'][dept] <= self.geometry_config.s_max * sqrt_area, name=f"width_ub[{dept}]")
            self.model.addConstr(self.variables['height'][dept] >= self.geometry_config.s_min * sqrt_area, name=f"height_lb[{dept}]")
            self.model.addConstr(self.variables['height'][dept] <= self.geometry_config.s_max * sqrt_area, name=f"height_ub[{dept}]")
            
            # Perimeter bounds
            if self.geometry_config.use_perimeter:
                self.model.addConstr(2 * (self.variables['width'][dept] + self.variables['height'][dept]) >= perim_min, name=f"perim_lb[{dept}]")
                self.model.addConstr(2 * (self.variables['width'][dept] + self.variables['height'][dept]) <= perim_max, name=f"perim_ub[{dept}]")
            
            # Aspect ratio bounds
            if self.geometry_config.use_aspect:
                self.model.addConstr(self.variables['width'][dept] <= self.geometry_config.aspect_ratio_limit * self.variables['height'][dept], name=f"aspect1[{dept}]")
                self.model.addConstr(self.variables['height'][dept] <= self.geometry_config.aspect_ratio_limit * self.variables['width'][dept], name=f"aspect2[{dept}]")
    
    def _add_distance_constraints(self) -> None:
        """Add Manhattan distance constraints."""
        unordered_pairs = [(i, j) for i, j in it.combinations(self.problem_data.dept_list, 2)]
        
        for (i, j) in unordered_pairs:
            # X distance envelope
            delta_x = self.variables['center_x'][i] - self.variables['center_x'][j]
            self.model.addConstr(self.variables['dist_x'][i, j] >= delta_x, name=f"distx_pos[{i},{j}]")
            self.model.addConstr(self.variables['dist_x'][i, j] >= -delta_x, name=f"distx_neg[{i},{j}]")
            
            # Y distance envelope
            delta_y = self.variables['center_y'][i] - self.variables['center_y'][j]
            self.model.addConstr(self.variables['dist_y'][i, j] >= delta_y, name=f"disty_pos[{i},{j}]")
            self.model.addConstr(self.variables['dist_y'][i, j] >= -delta_y, name=f"disty_neg[{i},{j}]")
    
    def _add_separation_constraints(self) -> None:
        """Add non-overlap separation constraints."""
        I = self.problem_data.dept_list
        unordered_pairs = [(i, j) for i, j in it.combinations(I, 2)]
        
        for i in I:
            for j in I:
                if i == j:
                    continue
                    
                cx, cy = self._get_clearance(i, j)
                
                # East separation indicator
                self.model.addGenConstrIndicator(
                    self.variables['is_east_of'][i, j], 1,
                    self.variables['x_east'][j] + cx <= self.variables['x_west'][i],
                    name=f"east_sep[{i}|{j}]"
                )
                
                # North separation indicator
                self.model.addGenConstrIndicator(
                    self.variables['is_north_of'][i, j], 1,
                    self.variables['y_north'][j] + cy <= self.variables['y_south'][i],
                    name=f"north_sep[{i}|{j}]"
                )
        
        # Coverage constraints
        for (i, j) in unordered_pairs:
            self.model.addConstr(
                self.variables['is_east_of'][i, j] + 
                self.variables['is_east_of'][j, i] + 
                self.variables['is_north_of'][i, j] + 
                self.variables['is_north_of'][j, i] >= 1,
                name=f"cover_sep[{i},{j}]"
            )
    
    def _apply_design_skeleton(self) -> None:
        """Apply design skeleton constraints."""
        if not self.problem_data.design_skeleton:
            return
            
        for i, rel, j in self.problem_data.design_skeleton:
            r = rel.strip().lower()
            if r in {"e", "east"}:
                self.model.addConstr(self.variables['is_east_of'][i, j] == 1, name=f"skel_east[{i}|{j}]")
                self.model.addConstr(self.variables['is_east_of'][j, i] == 0, name=f"skel_east_rev0[{j}|{i}]")
            elif r in {"w", "west"}:
                self.model.addConstr(self.variables['is_east_of'][j, i] == 1, name=f"skel_west[{i}|{j}]")
                self.model.addConstr(self.variables['is_east_of'][i, j] == 0, name=f"skel_west_rev0[{i}|{j}]")
            elif r in {"n", "north"}:
                self.model.addConstr(self.variables['is_north_of'][i, j] == 1, name=f"skel_north[{i}|{j}]")
                self.model.addConstr(self.variables['is_north_of'][j, i] == 0, name=f"skel_north_rev0[{j}|{i}]")
            elif r in {"s", "south"}:
                self.model.addConstr(self.variables['is_north_of'][j, i] == 1, name=f"skel_south[{i}|{j}]")
                self.model.addConstr(self.variables['is_north_of'][i, j] == 0, name=f"skel_south_rev0[{i}|{j}]")
    
    def _apply_anchors(self) -> None:
        """Apply anchor constraints."""
        if not self.problem_data.anchors:
            return
            
        for d, spec in self.problem_data.anchors.items():
            if d not in self.problem_data.dept_list:
                continue
                
            side = str(spec.get("side", "")).lower()
            offset = float(spec.get("offset", 0.0))
            
            # Clip offsets
            if side in {"south", "north"}:
                offset = max(0.0, min(offset, self.geometry_config.building_y))
            elif side in {"west", "east"}:
                offset = max(0.0, min(offset, self.geometry_config.building_x))
            
            if side == "south":
                self.model.addConstr(self.variables['y_south'][d] == offset, name=f"anchor_south[{d}]")
            elif side == "north":
                self.model.addConstr(self.variables['y_north'][d] == self.geometry_config.building_y - offset, name=f"anchor_north[{d}]")
            elif side == "west":
                self.model.addConstr(self.variables['x_west'][d] == offset, name=f"anchor_west[{d}]")
            elif side == "east":
                self.model.addConstr(self.variables['x_east'][d] == self.geometry_config.building_x - offset, name=f"anchor_east[{d}]")
    
    def _apply_fixed_departments(self) -> None:
        """Apply fixed department constraints."""
        if not self.problem_data.fixed_departments:
            return
            
        for d, spec in self.problem_data.fixed_departments.items():
            if d not in self.problem_data.dept_list:
                continue
                
            # Fix coordinates if provided
            for coord in ["x_west", "x_east", "y_south", "y_north"]:
                if coord in spec:
                    self.model.addConstr(self.variables[coord][d] == float(spec[coord]), name=f"fix_{coord}[{d}]")
            
            # Fix centers and dimensions if provided
            for param in ["center_x", "center_y", "width", "height"]:
                if param in spec:
                    self.model.addConstr(self.variables[param][d] == float(spec[param]), name=f"fix_{param}[{d}]")
    
    def _set_objective(self) -> None:
        """Set the objective function."""
        unordered_pairs = [(i, j) for i, j in it.combinations(self.problem_data.dept_list, 2)]
        
        objective = gp.quicksum(
            float(self.problem_data.weights.loc[i, j]) * 
            (self.variables['dist_x'][i, j] + self.variables['dist_y'][i, j]) 
            for (i, j) in unordered_pairs
        )
        self.model.setObjective(objective, GRB.MINIMIZE)
    
    def _get_clearance(self, i: str, j: str) -> Tuple[float, float]:
        """Get clearance between two departments."""
        if self.problem_data.pair_clearances and (i, j) in self.problem_data.pair_clearances:
            return self.problem_data.pair_clearances[(i, j)]
        if self.problem_data.pair_clearances and (j, i) in self.problem_data.pair_clearances:
            return self.problem_data.pair_clearances[(j, i)]
        return self.geometry_config.default_clearance, self.geometry_config.default_clearance
    
    def _get_skip_surrogates_dict(self, fixed_set: set) -> Dict[str, bool]:
        """Determine which departments should skip surrogate constraints."""
        def _width_fixed(spec: Dict[str, float]) -> bool:
            return ("width" in spec) or (("x_west" in spec) and ("x_east" in spec))
        
        def _height_fixed(spec: Dict[str, float]) -> bool:
            return ("height" in spec) or (("y_south" in spec) and ("y_north" in spec))
        
        skip_surrogates = {}
        for i in self.problem_data.dept_list:
            if i in fixed_set:
                spec = self.problem_data.fixed_departments[i]
                width_fixed = _width_fixed(spec)
                height_fixed = _height_fixed(spec)
                skip_surrogates[i] = width_fixed and height_fixed
            else:
                skip_surrogates[i] = False
        
        return skip_surrogates


class MIQCPSolver(MILPSolver):
    """Mixed Integer Quadratically Constrained Programming solver."""
    
    def __init__(self, problem_data: ProblemData, geometry_config: GeometryConfig, 
                 solver_config: SolverConfig):
        super().__init__(problem_data, geometry_config, solver_config)
    
    def build_model(self) -> None:
        """Build MIQCP model with exact area constraints."""
        # Force exact area calculation for MIQCP
        self.geometry_config.area_calculation = "exact"
        super().build_model()
        self.model.ModelName = "block_layout_miqcp"
    
    def _add_shape_constraints(self, dept: str, sqrt_area: float, perim_min: float, perim_max: float) -> None:
        """Add exact quadratic area constraints."""
        # Add exact area constraint using quadratic constraint
        area = self.problem_data.areas_m2[dept]
        self.model.addQConstr(
            self.variables['width'][dept] * self.variables['height'][dept] == float(area),
            name=f"area_exact[{dept}]"
        )
        
        # Still add basic bounds and aspect ratio constraints
        if self.geometry_config.use_aspect:
            self.model.addConstr(
                self.variables['width'][dept] <= self.geometry_config.aspect_ratio_limit * self.variables['height'][dept],
                name=f"aspect1[{dept}]"
            )
            self.model.addConstr(
                self.variables['height'][dept] <= self.geometry_config.aspect_ratio_limit * self.variables['width'][dept],
                name=f"aspect2[{dept}]"
            )


class SolverFactory:
    """Factory class for creating solvers."""
    
    @staticmethod
    def create_solver(solver_type: str, problem_data: ProblemData, 
                     geometry_config: GeometryConfig, solver_config: SolverConfig) -> OptimizationSolver:
        """Create solver based on type."""
        if solver_type.lower() == "milp":
            return MILPSolver(problem_data, geometry_config, solver_config)
        elif solver_type.lower() == "miqcp":
            return MIQCPSolver(problem_data, geometry_config, solver_config)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")


def build_block_layout_model(
    dept_list: List[str],
    areas_m2: Dict[str, float],
    weights: pd.DataFrame,
    building_x: float,
    building_y: float,
    *,
    # Shape controls
    s_min: float = 0.7,
    s_max: float = 1.6,
    use_perimeter: bool = True,
    rho_perimeter: float = 1.30,
    use_aspect: bool = True,
    aspect_ratio_limit: float = 3.0,
    # Clearances
    default_clearance: float = 0.0,
    pair_clearances: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    # Anchors and design
    anchors: Optional[Dict[str, Dict[str, float]]] = None,
    design_skeleton: Optional[List[Tuple[str, str, str]]] = None,
    fixed_departments: Optional[Dict[str, Dict[str, float]]] = None,
    # Solver controls
    mip_gap: float = 0.02,
    time_limit: Optional[int] = None,
    log_to_console: bool = True,
    area_band_pct: Optional[float] = None,
    area_calculation: Literal["exact", "envelope"] = "envelope",
    write_lp: str = "data/lp/block_layout.lp"
) -> Tuple[gp.Model, dict]:
    """
    Wrapper function to maintain compatibility with original function signature.
    Creates appropriate solver and returns model and variables.
    """
    # Create configuration objects
    problem_data = ProblemData(
        dept_list=dept_list,
        areas_m2=areas_m2,
        weights=weights,
        pair_clearances=pair_clearances,
        anchors=anchors,
        design_skeleton=design_skeleton,
        fixed_departments=fixed_departments
    )
    
    geometry_config = GeometryConfig(
        building_x=building_x,
        building_y=building_y,
        s_min=s_min,
        s_max=s_max,
        use_perimeter=use_perimeter,
        rho_perimeter=rho_perimeter,
        use_aspect=use_aspect,
        aspect_ratio_limit=aspect_ratio_limit,
        default_clearance=default_clearance,
        area_calculation=area_calculation,
        area_band_pct=area_band_pct
    )
    
    solver_config = SolverConfig(
        mip_gap=mip_gap,
        time_limit=time_limit,
        log_to_console=log_to_console,
        write_lp=write_lp
    )
    
    # Determine solver type based on area calculation
    solver_type = "miqcp" if area_calculation == "exact" else "milp"
    
    # Create and build solver
    solver = SolverFactory.create_solver(solver_type, problem_data, geometry_config, solver_config)
    solver.build_model()
    
    return solver.model, solver.variables


# Example usage
if __name__ == "__main__":
    # Example problem setup
    dept_list = ["A", "B", "C"]
    areas_m2 = {"A": 100, "B": 150, "C": 120}
    weights = pd.DataFrame([[0, 5, 3], [5, 0, 8], [3, 8, 0]], 
                          index=dept_list, columns=dept_list)
    
    # Create configuration objects
    problem_data = ProblemData(dept_list, areas_m2, weights)
    geometry_config = GeometryConfig(building_x=50, building_y=40)
    solver_config = SolverConfig(log_to_console=True, mip_gap=0.01)
    
    # Create and solve with MILP
    milp_solver = SolverFactory.create_solver("milp", problem_data, geometry_config, solver_config)
    milp_results = milp_solver.solve()
    print("MILP Results:", milp_results['status'])
    
    # Create and solve with MIQCP
    miqcp_solver = SolverFactory.create_solver("miqcp", problem_data, geometry_config, solver_config)
    miqcp_results = miqcp_solver.solve()
    print("MIQCP Results:", miqcp_results['status'])
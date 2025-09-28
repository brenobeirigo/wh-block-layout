import os
import gurobipy as gp
from gurobipy import GRB
import math

import itertools as it
from typing import Dict, List, Optional, Tuple, Literal
import pandas as pd

import logging
logger = logging.getLogger(__name__)

def build_block_layout_model(
    dept_list: List[str],
    areas_m2: Dict[str, float],
    weights: pd.DataFrame,
    building_x: float,
    building_y: float,
    *,
    # Shape controls that approximate area without bilinear terms
    s_min: float = 0.7,
    s_max: float = 1.6,
    use_perimeter: bool = True,
    rho_perimeter: float = 1.30,
    use_aspect: bool = True,
    aspect_ratio_limit: float = 3.0,
    # Clearances
    default_clearance: float = 0.0,
    pair_clearances: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    # Anchors on walls (offset only). Any "span_max" is deliberately ignored.
    anchors: Optional[Dict[str, Dict[str, float]]] = None,
    # Design skeleton: fix directional relations a priori
    design_skeleton: Optional[List[Tuple[str, str, str]]] = None,
    # Fixed departments: freeze sides or center+size
    fixed_departments: Optional[Dict[str, Dict[str, float]]] = None,
    # Solver controls
    mip_gap: float = 0.02,
    time_limit: Optional[int] = None,
    log_to_console: bool = True,
    area_calculation: Literal["exact", "envelope"] = "envelope",
    write_lp: str = "data/lp/block_layout.lp"
) -> Tuple[gp.Model, dict]:
    """
    Build a Gurobi MILP for rectangular block layout with:
      1) nonoverlap via indicator binaries,
      2) closeness-weighted Manhattan distances,
      3) optional design skeleton (pre-fixed east/north relations),
      4) optional fixed departments (immutable rectangles or center+size),
      5) anchors that pin one side of a rectangle to a chosen wall.

    Parameters
    ----------
    dept_list : list of str
        Names of departments, the model index set I.
    areas_m2 : dict[str, float]
        Planned area for each department in square meters.
    weights : pandas.DataFrame
        Symmetric, nonnegative weight matrix W with rows and columns indexed by names in dept_list.
        Larger W[i,j] pulls centers of i and j closer in the objective.
    building_x : float
        Building size along the x direction in meters.
    building_y : float
        Building size along the y direction in meters.
    s_min : float, optional
        Lower factor for side bounds relative to sqrt(area), default 0.7.
    s_max : float, optional
        Upper factor for side bounds relative to sqrt(area), default 1.6.
    use_perimeter : bool, optional
        If True, adds bounds on perimeter 2(width+height) around the square perimeter.
    rho_perimeter : float, optional
        Upper perimeter multiplier. P_max = rho_perimeter * P_min.
    use_aspect : bool, optional
        If True, enforces max(width,height) <= aspect_ratio_limit * min(width,height).
    aspect_ratio_limit : float, optional
        Upper bound on aspect ratio R >= 1.
    default_clearance : float, optional
        Default minimum separation for the chosen separation axis.
    pair_clearances : dict[(str,str)->(float,float)], optional
        Optional ordered pair map (i,j) -> (clear_x, clear_y). If missing, reverse (j,i) is tried,
        else the default_clearance is used for both axes.
    anchors : dict[str, dict], optional
        Per department anchor with {"side": "south"|"north"|"west"|"east", "offset": float}.
        Pins exactly one edge to a wall line. Any "span_max" passed in is ignored to avoid conflicts.
    design_skeleton : list[(str,str,str)], optional
        List of directional relations to fix a priori. Each triple is (i, rel, j) with rel in
        {"east","west","north","south","E","W","N","S"}.
        Example: ("Receiving","east","Storage") or ("Packing","north","Assembly").
    fixed_departments : dict[str, dict], optional
        Map from department to fields that must be fixed by equality constraint. Any of:
          - sides:  "x_west","x_east","y_south","y_north"
          - center: "center_x","center_y"
          - size:   "width","height"
        If both width and height are fixed for a department, shape surrogate bounds are skipped for it.
    mip_gap : float, optional
        Relative gap for Gurobi termination, default 0.02.
    time_limit : int, optional
        Time limit in seconds. None means no time limit.
    log_to_console : bool, optional
        If True, print solver output to console.
    area_band_pct : float, optional
        Percentage band around the area for soft constraints, default 0.1.
    area_calculation : str, optional
        Method to calculate area for soft constraints, default "exact".
    write_lp : str, optional
        Path to write the LP file, default "data/lp/block_layout.lp".

    Returns
    -------
    model : gurobipy.Model
        The constructed model, ready for optimize().
    variables : dict
        Dictionary with all decision variables for later inspection or plotting:
        {
          "x_west","x_east","y_south","y_north",
          "width","height","center_x","center_y",
          "dist_x","dist_y",
          "is_east_of","is_north_of"
        }

    Notes
    -----
    Distances use linear absolute value envelopes. For each unordered pair (i,j),
    dist_x[i,j] >=  ±(center_x[i] − center_x[j]), and analogously for y.
    With W >= 0 and minimization, these envelopes become equalities at optimality.

    The design skeleton fixes chosen binaries, which drastically reduces branching
    and converts the search into a packing problem for the remaining degrees of freedom.

    Any span limits are intentionally removed to avoid common infeasibilities that
    arise when span limits contradict lower side bounds derived from area.
    """
    # ---------- Validate inputs ----------
    # Ensure weights contains all departments in both axes
    assert set(dept_list).issubset(set(weights.index)), "weights is missing some departments in rows"
    assert set(dept_list).issubset(set(weights.columns)), "weights is missing some departments in columns"

    # Copy the index set so we do not mutate the caller's list
    I = list(dept_list)

    # All unordered pairs i<j, used for distance and separation coverage
    unordered_pairs = [(i, j) for i, j in it.combinations(I, 2)]

    # Utility to read W_ij as float
    def W(i: str, j: str) -> float:
        return float(weights.loc[i, j])

    # ---------- Process fixed departments metadata ----------
    fixed_departments = fixed_departments or {}
    fixed_set = set(fixed_departments.keys())

    # A small helper to detect whether width or height will be fixed by the user spec
    def _width_fixed(spec: Dict[str, float]) -> bool:
        return ("width" in spec) or (("x_west" in spec) and ("x_east" in spec))

    def _height_fixed(spec: Dict[str, float]) -> bool:
        return ("height" in spec) or (("y_south" in spec) and ("y_north" in spec))

    # Booleans to decide whether to skip surrogate bounds for a given department
    width_is_fixed  = {i: (_width_fixed(fixed_departments[i])  if i in fixed_set else False) for i in I}
    height_is_fixed = {i: (_height_fixed(fixed_departments[i]) if i in fixed_set else False) for i in I}
    skip_surrogates = {i: (width_is_fixed[i] and height_is_fixed[i]) for i in I} 

    # ---------- Create model and set solver parameters ----------
    model = gp.Model("block_layout")
    model.Params.OutputFlag = 1 if log_to_console else 0     # 1 prints Gurobi log, 0 is silent
    if time_limit is not None:
        model.Params.TimeLimit = time_limit                  # seconds
    model.Params.MIPGap = mip_gap                            # target relative optimality gap

    # ---------- Decision variables ----------
    # Rectangle sides for each department i
    x_west  = model.addVars(I, lb=0.0, ub=building_x, name="x_west")   # left edge
    x_east  = model.addVars(I, lb=0.0, ub=building_x, name="x_east")   # right edge
    y_south = model.addVars(I, lb=0.0, ub=building_y, name="y_south")  # bottom edge
    y_north = model.addVars(I, lb=0.0, ub=building_y, name="y_north")  # top edge

    # Width and height derived from sides (kept as explicit variables for readability and constraints)
    width   = model.addVars(I, lb=0.0, name="width")
    height  = model.addVars(I, lb=0.0, name="height")

    # Rectangle centers used in the distance objective
    center_x = model.addVars(I, lb=0.0, ub=building_x, name="center_x")
    center_y = model.addVars(I, lb=0.0, ub=building_y, name="center_y")

    # Distance envelopes for unordered pairs (i<j)
    dist_x = model.addVars(unordered_pairs, lb=0.0, name="dist_x")
    dist_y = model.addVars(unordered_pairs, lb=0.0, name="dist_y")

    # Indicator binaries for ordered pairs (i,j)
    # is_east_of[i,j] = 1 means i is strictly east of j
    # is_north_of[i,j] = 1 means i is strictly north of j
    is_east_of  = model.addVars(I, I, vtype=GRB.BINARY, name="is_east_of")
    is_north_of = model.addVars(I, I, vtype=GRB.BINARY, name="is_north_of")

    # ---------- Geometry definitions and simple bounds ----------
    for i in I:
        # Define width and height exactly from the side coordinates
        model.addConstr(width[i]  == x_east[i]  - x_west[i],  name=f"def_width[{i}]")
        model.addConstr(height[i] == y_north[i] - y_south[i], name=f"def_height[{i}]")

        # Define centers as midpoints of opposite sides
        model.addConstr(center_x[i] == 0.5 * (x_west[i] + x_east[i]),  name=f"def_center_x[{i}]")
        model.addConstr(center_y[i] == 0.5 * (y_south[i] + y_north[i]), name=f"def_center_y[{i}]")

        # Maintain ordering of sides, avoids negative width or height
        model.addConstr(x_west[i]  <= x_east[i],  name=f"x_order[{i}]")
        model.addConstr(y_south[i] <= y_north[i], name=f"y_order[{i}]")

        if not skip_surrogates[i]:
            
            if area_calculation == "exact":
                model.addQConstr(width[i] * height[i] == float(areas_m2[i]), name=f"area_exact[{i}]")
            
            else:

                # Precompute sqrt(area) and perimeter guides for each department
                sqrt_area = {i: math.sqrt(areas_m2[i]) for i in I}       # used to bound sides linearly
                perim_min = {i: 4.0 * sqrt_area[i] for i in I}           # square perimeter: 4*sqrt(area)
                perim_max = {i: rho_perimeter * perim_min[i] for i in I} # relaxed upper perimeter

                
                # Side bounds around sqrt(area) keep shapes reasonable without w*h = area
                model.addConstr(width[i]  >= s_min * sqrt_area[i], name=f"width_lb[{i}]")
                model.addConstr(width[i]  <= s_max * sqrt_area[i], name=f"width_ub[{i}]")
                model.addConstr(height[i] >= s_min * sqrt_area[i], name=f"height_lb[{i}]")
                model.addConstr(height[i] <= s_max * sqrt_area[i], name=f"height_ub[{i}]")

                # Optional perimeter bounds tighten area approximation
                if use_perimeter:
                    model.addConstr(2 * (width[i] + height[i]) >= perim_min[i], name=f"perim_lb[{i}]")
                    model.addConstr(2 * (width[i] + height[i]) <= perim_max[i], name=f"perim_ub[{i}]")

                # Optional aspect ratio control
                if use_aspect:
                    model.addConstr(width[i]  <= aspect_ratio_limit * height[i], name=f"aspect1[{i}]")
                    model.addConstr(height[i] <= aspect_ratio_limit * width[i],  name=f"aspect2[{i}]")
            
        # Keep all sides inside the building envelope
        model.addConstr(x_west[i]  >= 0.0)
        model.addConstr(y_south[i] >= 0.0)
        model.addConstr(x_east[i]  <= building_x)
        model.addConstr(y_north[i] <= building_y)

    # ---------- Absolute value envelopes for Manhattan distances ----------
    # For each unordered pair (i,j), we create two inequalities that together imply
    # dist_x[i,j] >= |center_x[i] - center_x[j]|, similarly for y.
    for (i, j) in unordered_pairs:
        delta_x = center_x[i] - center_x[j]                      # Δx between centers
        model.addConstr(dist_x[i, j] >=  delta_x, name=f"distx_pos[{i},{j}]")  # dist_x >= +Δx
        model.addConstr(dist_x[i, j] >= -delta_x, name=f"distx_neg[{i},{j}]")  # dist_x >= -Δx

        delta_y = center_y[i] - center_y[j]                      # Δy between centers
        model.addConstr(dist_y[i, j] >=  delta_y, name=f"disty_pos[{i},{j}]")  # dist_y >= +Δy
        model.addConstr(dist_y[i, j] >= -delta_y, name=f"disty_neg[{i},{j}]")  # dist_y >= -Δy

    # ---------- Nonoverlap via indicator constraints with clearances ----------
    # Helper that returns the ordered pair clearance (clear_x, clear_y).
    def _clearance(i: str, j: str) -> Tuple[float, float]:
        if pair_clearances and (i, j) in pair_clearances:
            return pair_clearances[(i, j)]
        if pair_clearances and (j, i) in pair_clearances:
            return pair_clearances[(j, i)]
        return default_clearance, default_clearance

    for i in I:
        for j in I:
            if i == j:
                continue
            cx, cy = _clearance(i, j)  # requested separation along each axis

            # If is_east_of[i,j] = 1 then x_east[j] + cx <= x_west[i]
            # This forces the entire rectangle i to be strictly to the east of j with a clearance.
            model.addGenConstrIndicator(
                is_east_of[i, j], 1, x_east[j] + cx <= x_west[i],
                name=f"east_sep[{i}|{j}]"
            )

            # If is_north_of[i,j] = 1 then y_north[j] + cy <= y_south[i]
            # This forces rectangle i to be strictly to the north of j with a clearance.
            model.addGenConstrIndicator(
                is_north_of[i, j], 1, y_north[j] + cy <= y_south[i],
                name=f"north_sep[{i}|{j}]"
            )

    # For each unordered pair, at least one directional separation must hold.
    # The solver can choose east or north (or even both) to prevent overlap.
    for (i, j) in unordered_pairs:
        model.addConstr(
            is_east_of[i, j] 
            + is_east_of[j, i] 
            + is_north_of[i, j] 
            + is_north_of[j, i] 
            >= 1,
            name=f"cover_sep[{i},{j}]"
        )

    # ---------- Apply design skeleton: fix chosen z binaries ahead of time ----------
    # This is where you pre-impose relations like "Receiving east of Storage".
    if design_skeleton:
        for i, rel, j in design_skeleton:
            if (i not in I) or (j not in I):
                raise ValueError(f"Design skeleton references unknown departments: {(i, j)} - {I}")
            r = rel.strip().lower()
            if r in {"e", "east"}:
                model.addConstr(is_east_of[i, j] == 1, name=f"skel_east[{i}|{j}]")
                model.addConstr(is_east_of[j, i] == 0, name=f"skel_east_rev0[{j}|{i}]")
            elif r in {"w", "west"}:
                model.addConstr(is_east_of[j, i] == 1, name=f"skel_west[{i}|{j}]")   # j east of i
                model.addConstr(is_east_of[i, j] == 0, name=f"skel_west_rev0[{i}|{j}]")
            elif r in {"n", "north"}:
                model.addConstr(is_north_of[i, j] == 1, name=f"skel_north[{i}|{j}]")
                model.addConstr(is_north_of[j, i] == 0, name=f"skel_north_rev0[{j}|{i}]")
            elif r in {"s", "south"}:
                model.addConstr(is_north_of[j, i] == 1, name=f"skel_south[{i}|{j}]") # j north of i
                model.addConstr(is_north_of[i, j] == 0, name=f"skel_south_rev0[{i}|{j}]")
            else:
                raise ValueError(f"Unknown skeleton relation '{rel}' for ({i},{j}).")

    # ---------- Anchors: pin one edge to a wall at a given offset (no span limits) ----------
    if anchors:
        for d, spec in anchors.items():
            if d not in I:
                continue
            side   = str(spec.get("side", "")).lower()
            offset = float(spec.get("offset", 0.0))

            # Clip offsets into the building to avoid trivial infeasibility
            if side in {"south", "north"}:
                offset = max(0.0, min(offset, building_y))
            elif side in {"west", "east"}:
                offset = max(0.0, min(offset, building_x))

            # Ignore legacy span_max if present, provide a friendly note
            if "span_max" in spec:
                print(f"[info] Ignoring 'span_max' for '{d}' to avoid conflicts with side lower bounds.")

            if side == "south":
                model.addConstr(y_south[d] == offset, name=f"anchor_south[{d}]")
            elif side == "north":
                model.addConstr(y_north[d] == building_y - offset, name=f"anchor_north[{d}]")
            elif side == "west":
                model.addConstr(x_west[d] == offset, name=f"anchor_west[{d}]")
            elif side == "east":
                model.addConstr(x_east[d] == building_x - offset, name=f"anchor_east[{d}]")
            else:
                # Unrecognized side value, do nothing
                pass

    # ---------- Fixed departments: freeze any provided fields by equality ----------
    if fixed_departments:
        for d, spec in fixed_departments.items():
            if d not in I:
                continue
            # Fix sides if provided
            if "x_west"  in spec: model.addConstr(x_west[d]  == float(spec["x_west"]),  name=f"fix_x_west[{d}]")
            if "x_east"  in spec: model.addConstr(x_east[d]  == float(spec["x_east"]),  name=f"fix_x_east[{d}]")
            if "y_south" in spec: model.addConstr(y_south[d] == float(spec["y_south"]), name=f"fix_y_south[{d}]")
            if "y_north" in spec: model.addConstr(y_north[d] == float(spec["y_north"]), name=f"fix_y_north[{d}]")
            # Fix centers or sizes if provided
            if "center_x" in spec: model.addConstr(center_x[d] == float(spec["center_x"]), name=f"fix_center_x[{d}]")
            if "center_y" in spec: model.addConstr(center_y[d] == float(spec["center_y"]), name=f"fix_center_y[{d}]")
            if "width"    in spec: model.addConstr(width[d]    == float(spec["width"]),    name=f"fix_width[{d}]")
            if "height"   in spec: model.addConstr(height[d]   == float(spec["height"]),   name=f"fix_height[{d}]")

    # ---------- Objective: minimize closeness-weighted sum of L1 distances ----------
    objective = gp.quicksum(
        W(i, j) * (dist_x[i, j] + dist_y[i, j]) for (i, j) in unordered_pairs
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # ---------- Pack variables for downstream use ----------
    variables = dict(
        x_west=x_west, x_east=x_east, y_south=y_south, y_north=y_north,
        width=width, height=height, center_x=center_x, center_y=center_y,
        dist_x=dist_x, dist_y=dist_y, is_east_of=is_east_of, is_north_of=is_north_of
    )
    
    # Save Gurobi mip model
    if write_lp:
        dir_path = os.path.dirname(write_lp)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        model.write(write_lp)
    
    return model, variables
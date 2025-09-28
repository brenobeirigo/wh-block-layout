from whlayout.model import build_block_layout_model              
from whlayout.process import map_weights                         
from whlayout.layoututils import propose_building_sides          
from whlayout.utils import extract_solution_table                
from whlayout.plot import plot_layout
from whlayout.io import load_data                          

import pandas as pd


if __name__ == "__main__":

    # -----------------------------
    # Problem data (areas & A/R)
    # -----------------------------
    # Planned areas (m²).
    df_areas = load_data("instances/17_depts/space_requirements.csv")

    # Adjacency/closeness ratings [-2, 2]
    df_adj_matrix = load_data("instances/17_depts/adjacency_matrix.csv")

    # --------------------------------
    #  Experiment: Adding fixed stairs
    # --------------------------------
    df_adj_matrix["Stairs"] =  0
    df_adj_matrix.loc["Stairs"] = 0
    df_areas.at["Stairs", "Planned Space (sq m)"] = 15

    # Suppose the designer wants to fix the stairs
    fixed_departments = {}
    fixed_departments["Stairs"] = {
        "width": 3,
        "height": 5,
        "center_x": 1.5,
        "center_y": 100,
    }

    # Check
    print("Areas (m²):\n", df_areas)
    print("\nCloseness ratings:\n", df_adj_matrix)

    # Names and areas for convenience
    departments = df_areas.index.tolist()
    areas_m2 = df_areas["Planned Space (sq m)"].to_dict()
    print(f"\nTotal planned area (m²): {sum(areas_m2.values()):.1f}")

    # ----------------------------------------
    # Convert ratings -> nonnegative weights
    # ----------------------------------------
    W = map_weights(ratings=df_adj_matrix, scheme="exp")
    print("\nWeights (exp-mapped):\n", W.head())

    # -----------------------------------------------------
    # Building size suggestion
    # -----------------------------------------------------
    Bx, By = propose_building_sides(areas_m2, slack=1.3, aspect_ratio=1)
    print(f"\nSuggested building (Bx, By) = ({Bx}, {By})")

    # ----------------------------------------------------
    # U-shape intent via anchors
    # ----------------------------------------------------
    # Anchors pin one edge to a wall
    anchors = {
        "Receiving": {"side": "south", "offset": 0.0},
        "Shipping":  {"side": "south", "offset": 0.0},
    }
    
    
    anchors["Stairs"] = {"side": "west", "offset": 0.0}

    # -------------------------------
    # Build and solve the MILP
    # -------------------------------
    # Key modeling choices for this example:
    # - area_model="envelope"
    model, var = build_block_layout_model(
        dept_list=departments,
        areas_m2=areas_m2,
        weights=W,
        building_x=Bx,
        building_y=By,
        # layout intent
        anchors=anchors,
        fixed_departments=fixed_departments,
        # area modeling
        area_calculation="envelope",
        # geometry & degrees of freedom
        s_min=0,
        s_max=4,
        use_perimeter=True,
        rho_perimeter=1.3,
        use_aspect=True,
        aspect_ratio_limit=4.0,
        # solver
        mip_gap=0.2,
        time_limit=None,
        log_to_console=True,
        # useful for debugging
        write_lp="data/lp/u_shape_milp_17_depts.lp",
    )

    model.optimize()

    # -----------------------------------
    # Extract and sanity-check results
    # -----------------------------------
    if model.SolCount == 0:
        raise RuntimeError(
            "No feasible solution found."
            "Increase building slack, relax skeleton, or reduce clearances.")

    obj = model.ObjVal
    print(f"\nOptimal objective value (weighted L1): {obj:.2f}")

    # Pull a tidy solution table (one row per department)
    df_sol = extract_solution_table(
        dept_list=departments,
        building_x=Bx,
        building_y=By,
        vars_out=var,
        # lets the extractor compute w*h and area error if implemented
        areas_m2=areas_m2,
    )
    # Helpful view for quick grading / debugging
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print("\nSolution (geometry):\n", df_sol)

    # --------------------------------
    # Visual inspection (matplotlib)
    # --------------------------------
    # Plot with labels; turn on annotate_area to print ~area on each box if your plotter supports it
    plot_layout(
        df_solution=df_sol.rename(
            columns={
                "xW": "x_west", "xE": "x_east",
                "yS": "y_south", "yN": "y_north",
                "w": "width", "h": "height",
                "ax": "center_x", "by": "center_y",
                "Area (approx)": "area_approx",
            }
        ),
        building_x=Bx,
        building_y=By,
        title=f"U-shape Layout (obj={obj:.2f}, {Bx}x{By} m)",
        label_centers=True,
        label_boxes=True,
        annotate_area=True,
        fontsize=10,
        linewidth=2.0,
    )

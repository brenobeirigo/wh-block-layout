from whlayout.model import build_block_layout_model              
from whlayout.process import map_weights                         
from whlayout.layoututils import propose_building_sides          
from whlayout.utils import extract_solution_table                
from whlayout.plot import plot_layout                           

import numpy as np
import pandas as pd


if __name__ == "__main__":

    # -----------------------------
    # Problem data (areas & A/R)
    # -----------------------------
    # Planned areas (m²)
    df_areas = (
        pd.DataFrame(
            {
                "Department": [
                    "Receiving",
                    "Staging",
                    "Pallet Storage",
                    "Case Picking",
                    "Shipping",
                    "Offices",
                ],
                # Example areas in m²
                "Planned Space (sq m)": [
                    1000,
                    1000,
                    2500,
                    2500,
                    1000,
                    100
                ],
            }
        )
        .set_index("Department")
    )
    print("Areas (m²):\n", df_areas)

    # Adjacency/closeness ratings [-2, 2]
    # Receiving is close to Staging and Shipping
    # Pallet Storage and Case Picking are close to Staging
    # Offices are to be isolated from the main flow
    df_adj_matrix = pd.DataFrame(
        {
            "Receiving":      [np.nan,  2,  1,  0,  2,  0],
            "Staging":        [2,     np.nan, 2,  1,  2,  0],
            "Pallet Storage": [1,       2,  np.nan, 2,  1,  -2],
            "Case Picking":   [0,       1,  2,  np.nan, 2,  -2],
            "Shipping":       [2,       2,  1,  2,  np.nan, 0],
            "Offices":        [0,       0,  -2,  -2,  0,  np.nan],
        },
        index=df_areas.index,
    )
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
    Bx, By = propose_building_sides(areas_m2, slack=1.0, aspect_ratio=1)
    print(f"\nSuggested building (Bx, By) = ({Bx}, {By})")

    # ----------------------------------------------------
    # U-shape intent via anchors
    # ----------------------------------------------------
    # Anchors pin one edge to a wall
    anchors = {
        "Receiving": {"side": "south", "offset": 0.0},
        "Shipping":  {"side": "south", "offset": 0.0},
    }

    # The skeleton fixes chosen east/north relations a priori to guide the flow.
    # Keep this *minimal*; overconstraining invites infeasibility.
    design_skeleton = [
        ("Staging",        "north", "Receiving"),      # staging behind inbound
        ("Staging",        "north", "Shipping"),       # and behind outbound
        ("Pallet Storage", "north", "Staging"),        # deep storage at the back
        ("Case Picking", "north", "Staging"),        # deep storage at the back
        ("Case Picking",   "east",  "Pallet Storage"), # picking to the east
        ("Offices",        "east",  "Shipping"),       # offices out of the main flow
        ("Shipping",       "east",  "Receiving"),
    ]

    # ---------------------------------------------------
    # Clearances (aisles/corridors) between key pairs
    # ---------------------------------------------------
    # Ordered-pair (i, j) -> (clear_x, clear_y).
    # The clearance applies *along the axis chosen* by the indicator that separates them.
    pair_clearances = {
        ("Receiving", "Shipping"): (6.0, 0.0),  # buffer between dock blocks (east–west corridor)
        ("Staging",   "Receiving"): (0.0, 3.0), # aisle north of receiving
        ("Staging",   "Shipping"):  (0.0, 3.0),
        ("Pallet Storage", "Staging"): (0.0, 4.0),
        ("Case Picking",   "Pallet Storage"): (0.0, 3.0),
        ("Case Picking",   "Shipping"):       (0.0, 3.0),
        ("Offices",        "Shipping"):       (5.0, 0.0),  # isolate offices
    }

    fixed_departments = {
        "Staging": {
            "width": 100,
            "height": 10,
        }
    }
    
    # -------------------------------
    # Build and solve the MIP/MIQCP
    # -------------------------------
    # Key modeling choices for this example:
    # - area_model="exact": exact product w_i * h_i = A_i (MIQCP with NonConvex=2)
    model, var = build_block_layout_model(
        dept_list=departments,
        areas_m2=areas_m2,
        weights=W,
        building_x=Bx,
        building_y=By,
        # layout intent
        anchors=anchors,
        # design_skeleton=design_skeleton,
        # pair_clearances=pair_clearances,
        default_clearance=0.0,    # used when a pair is not in pair_clearances
        # fixed_departments=fixed_departments,
        # area modeling
        area_calculation="exact",
        # solver
        mip_gap=0.02,
        time_limit=None,
        log_to_console=True,
        write_lp="data/lp/u_shape_exact.lp",  # useful for debugging
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

from typing import Dict, List, Tuple, Optional
import itertools as it
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _val(x) -> float:
    """
    Safe read of a Gurobi Var value.

    Returns x.X if available, otherwise numpy.nan.
    """
    try:
        return float(x.X)
    except Exception:
        return float("nan")


def _pair_val(td, i: str, j: str) -> float:
    """
    Safe read of a Gurobi tupledict entry td[i, j].

    Returns td[i, j].X if available, otherwise numpy.nan.
    """
    try:
        return float(td[i, j].X)
    except Exception:
        return float("nan")


def extract_solution_table(
    dept_list: List[str],
    building_x: float,
    building_y: float,
    vars_out: Dict[str, object],
    *,
    areas_m2: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a tidy per-department solution table.

    Parameters
    ----------
    dept_list : list of str
        Departments in the model, used to order rows.
    building_x : float
        Building size along x in meters. Saved as a convenience column.
    building_y : float
        Building size along y in meters. Saved as a convenience column.
    vars_out : dict
        Dictionary returned by the model builder. Expected keys:
        - "x_west","x_east","y_south","y_north"
        - "width","height","center_x","center_y"
    areas_m2 : dict[str, float], optional
        Planned areas for each department. If given, area errors are computed.

    Returns
    -------
    df : pandas.DataFrame
        Index by Department with columns:
        ["x_west","x_east","y_south","y_north",
         "width","height","center_x","center_y",
         "area_approx","aspect_ratio","perimeter",
         "building_x","building_y",
         "area_target","area_gap_abs","area_gap_pct"]  (last three only if areas_m2 given)

    Notes
    -----
    - area_approx equals width * height, which may differ from the target since the model
      uses linear surrogates for area.
    - aspect_ratio equals max(width,height) / min(width,height) with 1 when square
      and nan if any side is missing.
    """
    x_west   = vars_out["x_west"]
    x_east   = vars_out["x_east"]
    y_south  = vars_out["y_south"]
    y_north  = vars_out["y_north"]
    width    = vars_out["width"]
    height   = vars_out["height"]
    center_x = vars_out["center_x"]
    center_y = vars_out["center_y"]

    rows = []
    for d in dept_list:
        # Read all primitive values using the safe getters
        xw = _val(x_west[d])
        xe = _val(x_east[d])
        ys = _val(y_south[d])
        yn = _val(y_north[d])
        w  = _val(width[d])
        h  = _val(height[d])
        cx = _val(center_x[d])
        cy = _val(center_y[d])

        # Derived geometry
        area_approx = (w * h) if (np.isfinite(w) and np.isfinite(h)) else np.nan
        if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
            aspect_ratio = max(w, h) / min(w, h)
        else:
            aspect_ratio = np.nan
        perimeter = 2 * (w + h) if (np.isfinite(w) and np.isfinite(h)) else np.nan

        rec = dict(
            Department=d,
            x_west=xw, x_east=xe, y_south=ys, y_north=yn,
            width=w, height=h, center_x=cx, center_y=cy,
            area_approx=area_approx,
            aspect_ratio=aspect_ratio,
            perimeter=perimeter,
            building_x=building_x,
            building_y=building_y,
        )

        # Area diagnostics if targets are provided
        if areas_m2 and d in areas_m2:
            target = float(areas_m2[d])
            gap_abs = area_approx - target if np.isfinite(area_approx) else np.nan
            gap_pct = (gap_abs / target * 100.0) if (np.isfinite(gap_abs) and target > 0) else np.nan
            rec.update(
                area_target=target,
                area_gap_abs=gap_abs,
                area_gap_pct=gap_pct,
            )

        rows.append(rec)

    df = pd.DataFrame(rows).set_index("Department").sort_index()
    return df


def extract_pairwise_distances(
    dept_list: List[str],
    vars_out: Dict[str, object],
    *,
    weights: Optional[pd.DataFrame] = None,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a pairwise distance table for unordered pairs (i, j) with i < j.

    Parameters
    ----------
    dept_list : list of str
        Departments in the model, used to build pairs.
    vars_out : dict
        Dictionary returned by the model builder. Expected keys:
        - "center_x","center_y"
        - "dist_x","dist_y"
    weights : pandas.DataFrame, optional
        If provided, a 'weight_ij' column is added with W[i, j].
    atol : float, optional
        Tolerance used to report any envelope violation in 'envelope_violation'.

    Returns
    -------
    df : pandas.DataFrame
        Columns: ["i","j","center_x_i","center_y_i","center_x_j","center_y_j",
                  "dist_x","dist_y","manhattan","abs_dx_expected","abs_dy_expected",
                  "envelope_violation","weight_ij" (optional)]

    Notes
    -----
    - manhattan equals dist_x + dist_y. With a feasible optimal solution and W >= 0,
      the envelopes should be tight and manhattan equals abs_dx_expected + abs_dy_expected
      up to numerical tolerance.
    """
    center_x = vars_out["center_x"]
    center_y = vars_out["center_y"]
    dist_x   = vars_out["dist_x"]
    dist_y   = vars_out["dist_y"]

    rows = []
    for i, j in it.combinations(dept_list, 2):
        cxi = _val(center_x[i]); cyi = _val(center_y[i])
        cxj = _val(center_x[j]); cyj = _val(center_y[j])

        dx_ij = _pair_val(dist_x, i, j)
        dy_ij = _pair_val(dist_y, i, j)

        abs_dx = abs(cxi - cxj) if (np.isfinite(cxi) and np.isfinite(cxj)) else np.nan
        abs_dy = abs(cyi - cyj) if (np.isfinite(cyi) and np.isfinite(cyj)) else np.nan

        # Largest deviation between envelope variables and |delta|
        viol = max(
            0.0 if (np.isnan(dx_ij) or np.isnan(abs_dx)) else max(0.0, dx_ij - abs_dx),
            0.0 if (np.isnan(dy_ij) or np.isnan(abs_dy)) else max(0.0, dy_ij - abs_dy),
        )

        rec = dict(
            i=i, j=j,
            center_x_i=cxi, center_y_i=cyi,
            center_x_j=cxj, center_y_j=cyj,
            dist_x=dx_ij, dist_y=dy_ij,
            manhattan=(dx_ij + dy_ij) if (np.isfinite(dx_ij) and np.isfinite(dy_ij)) else np.nan,
            abs_dx_expected=abs_dx,
            abs_dy_expected=abs_dy,
            envelope_violation=(viol if viol > atol else 0.0),
        )
        if weights is not None:
            try:
                rec["weight_ij"] = float(weights.loc[i, j])
            except Exception:
                rec["weight_ij"] = np.nan

        rows.append(rec)

    df = pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)
    return df


def extract_relations(
    dept_list: List[str],
    vars_out: Dict[str, object],
) -> pd.DataFrame:
    """
    Report chosen nonoverlap relations for each unordered pair.

    Parameters
    ----------
    dept_list : list of str
        Departments in the model, used to build pairs.
    vars_out : dict
        Dictionary returned by the model builder. Expected keys:
        - "is_east_of","is_north_of" (binary tupledicts on ordered pairs)

    Returns
    -------
    df : pandas.DataFrame
        One row per unordered pair (i, j) with booleans indicating which
        direction is active:
        ["i","j","i_east_of_j","j_east_of_i","i_north_of_j","j_north_of_i",
         "has_east_sep","has_north_sep","covers_overlap"]

    Notes
    -----
    - covers_overlap equals True if at least one of the four directed binaries is 1.
      This is the logical coverage used to forbid overlap.
    """
    is_east_of  = vars_out["is_east_of"]
    is_north_of = vars_out["is_north_of"]

    rows = []
    for i, j in it.combinations(dept_list, 2):
        iE = bool(round(_pair_val(is_east_of,  i, j)))
        jE = bool(round(_pair_val(is_east_of,  j, i)))
        iN = bool(round(_pair_val(is_north_of, i, j)))
        jN = bool(round(_pair_val(is_north_of, j, i)))

        rows.append(dict(
            i=i, j=j,
            i_east_of_j=iE, j_east_of_i=jE,
            i_north_of_j=iN, j_north_of_i=jN,
            has_east_sep=(iE or jE),
            has_north_sep=(iN or jN),
            covers_overlap=(iE or jE or iN or jN),
        ))

    return pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)
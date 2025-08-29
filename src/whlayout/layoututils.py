import math
from typing import Dict, Tuple
import pandas as pd

def propose_building_sides(
    areas_m2: Dict[str, float],
    *,
    slack: float = 1.25,
    aspect_ratio: float = 1.0,
) -> Tuple[int, int]:
    """
    Suggest building sides (Bx, By) in meters from department areas.

    Parameters
    ----------
    areas_m2 : dict[str, float]
        Area of each department in m².
    slack : float, optional
        Multiplier applied to the total area to leave free space for layout optimization.
        For example, slack=1.25 adds 25% extra space. Default is 1.25.
    aspect_ratio : float, optional
        Target ratio Bx/By. Use 1.0 for a square building. Must be > 0. Default is 1.0.

    Returns
    -------
    (Bx, By) : tuple[int, int]
        Suggested integer building dimensions in meters.

    Notes
    -----
    Let T = slack × total area. To find the building dimensions:
    By = ceil(sqrt(T / aspect ratio)), and Bx = ceil(sqrt(T × aspect ratio)).
    """
    if not areas_m2:
        raise ValueError("areas_m2 is empty.")
    if slack <= 0:
        raise ValueError("slack must be > 0.")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be > 0.")

    total_area = sum(float(a) for a in areas_m2.values())
    if total_area <= 0:
        raise ValueError("Sum of areas must be > 0.")

    T = slack * total_area
    By = max(1, math.ceil(math.sqrt(T / aspect_ratio)))
    Bx = max(1, math.ceil(math.sqrt(T * aspect_ratio)))
    return Bx, By
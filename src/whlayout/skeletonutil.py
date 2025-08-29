from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def _first_match(names: List[str], keywords: List[str]) -> Optional[str]:
    """Return the first name containing any keyword (case-insensitive)."""
    low = {n.lower(): n for n in names}
    for nlow, orig in low.items():
        if any(k.lower() in nlow for k in keywords):
            return orig
    return None

def match_warehouse_roles(dept_list: List[str]) -> Dict[str, Optional[str]]:
    """
    Heuristically map canonical roles to your department names using substrings.
    Edit the keyword lists to fit your naming conventions.
    """
    roles = {
        "receiving":     _first_match(dept_list, ["receiving", "inbound", "dock in", "goods in"]),
        "shipping":      _first_match(dept_list, ["shipping", "outbound", "dispatch", "dock out"]),
        "staging":       _first_match(dept_list, ["staging", "stage", "buffer", "prestage"]),
        "qc":            _first_match(dept_list, ["quality", "qc", "inspection", "check"]),
        "storage":       _first_match(dept_list, ["storage", "reserve", "racking", "bulk"]),
        "forward_pick":  _first_match(dept_list, ["forward pick", "pick face", "fast pick", "flow rack"]),
        "picking":       _first_match(dept_list, ["picking", "picker", "order pick"]),
        "packing":       _first_match(dept_list, ["packing", "pack", "packout"]),
        "returns":       _first_match(dept_list, ["returns", "rtv", "reverse"]),
        "value_add":     _first_match(dept_list, ["vas", "value add", "kitting", "assembly"]),
        "offices":       _first_match(dept_list, ["office", "offices"]),
        "maintenance":   _first_match(dept_list, ["maintenance", "mro", "battery"]),
    }
    return roles

def make_ushape_inputs(
    dept_list: List[str],
    *,
    building_side: str = "south",
    dock_offset: float = 0.0,
    dock_min_gap_x: float = 3.0,   # min horizontal space between Receiving and Shipping if same wall
    aisle_clear_y: float = 2.0,    # min vertical clearance between docks and back-of-house
) -> Tuple[Dict[str, Dict[str, float]], List[Tuple[str,str,str]], Dict[Tuple[str,str], Tuple[float,float]], Dict[str, Optional[str]]]:
    """
    Construct logical U-shape inputs: anchors, design skeleton, and clearances.

    Returns
    -------
    anchors : dict
        {dept: {"side": building_side, "offset": dock_offset}}
    design_skeleton : list of (i, relation, j)
        Directional relations with 'east','west','north','south'.
    pair_clearances : dict
        {(i,j): (clear_x, clear_y)} used by the indicator constraints.
    roles : dict
        Which names were matched for canonical roles.
    """
    roles = match_warehouse_roles(dept_list)
    R = roles.get("receiving")
    S = roles.get("shipping")
    STG = roles.get("staging")
    QC = roles.get("qc")
    STR = roles.get("storage")
    FP = roles.get("forward_pick")
    PK = roles.get("picking")
    PCK = roles.get("packing")
    RET = roles.get("returns")

    anchors: Dict[str, Dict[str, float]] = {}
    skeleton: List[Tuple[str, str, str]] = []
    clearances: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # 1) U-shape wall anchors: Receiving and Shipping on the same wall
    if R:
        anchors[R] = {"side": building_side, "offset": dock_offset}
    if S:
        anchors[S] = {"side": building_side, "offset": dock_offset}

    # Shipping and receiving occupy the whole side. All departments are north of the dock.
    for dept in [R, S, STG, QC, STR, FP, PK, PCK, RET]:
        if dept and dept != S and dept != R:
            skeleton.append((dept, "north", R))
            skeleton.append((dept, "north", S))

    if R and S:
        skeleton.append((R, "west", S))
        # Keep a bit of horizontal room between the dock blocks
        clearances[(R, S)] = (dock_min_gap_x, 0.0)
        clearances[(S, R)] = (dock_min_gap_x, 0.0)

    # 3) Staging above Receiving (user preference)
    if STG and R:
        skeleton.append((STG, "north", R))
        clearances[(STG, R)] = (0.0, aisle_clear_y)

    # 4) Storage at the back: north of both dock blocks
    if STR:
        if R:
            skeleton.append((STR, "north", R))
            clearances[(STR, R)] = (0.0, aisle_clear_y)
        if S:
            skeleton.append((STR, "north", S))
            clearances[(STR, S)] = (0.0, aisle_clear_y)

    # 5) Forward-pick between storage and packing
    if FP:
        if STR:
            skeleton.append((STR, "north", FP))     # storage behind forward pick
            clearances[(STR, FP)] = (0.0, aisle_clear_y)
        if PCK:
            skeleton.append((FP, "north", PCK))     # forward pick above packing
            clearances[(FP, PCK)] = (0.0, aisle_clear_y)

    # 6) Picking near forward-pick (optional guidance)
    if PK and FP:
        skeleton.append((FP, "north", PK))          # forward pick above picking zone

    # 7) Packing near Shipping, between Receiving and Shipping
    if PCK:
        if R:
            skeleton.append((PCK, "east", R))       # to the right of Receiving
        if S:
            skeleton.append((PCK, "west", S))       # to the left of Shipping

    # 8) QC near Receiving
    if QC and R:
        skeleton.append((QC, "north", R))

    # 9) Returns near Shipping
    if RET and S:
        skeleton.append((RET, "north", S))

    return anchors, skeleton, clearances, roles
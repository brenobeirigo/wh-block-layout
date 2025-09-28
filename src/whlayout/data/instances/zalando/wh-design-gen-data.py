# generate_zalando_greenfield_case_csvs.py
# Greenfield Zalando-like fashion FC, 140,000 m² total
# Data reflects: KNAPP shuttle + pocket sorter, high returns (fashion), GOH + flat picking
# Outputs:
#   - zalando_space_program_abs.csv
#   - muther_legend.csv
#   - zalando_adjacency_matrix.csv
#   - zalando_adjacency_pairs_why.csv

import pandas as pd
from pathlib import Path

# -----------------------------
# 1) Space program (absolute m²)
# -----------------------------
sqm_by_area = {
    "Inbound Dock": 3500,
    "Receiving/Staging": 7000,
    "QA & Inspection": 2800,
    "Returns Intake & Triage": 6300,
    "Returns Refurb & Repack": 4200,
    "Cross-Dock": 2800,
    "VAS/Kitting": 3500,

    # Storage & engines
    "Pallet Reserve Storage (Bulk)": 26600,   # bulk long-tail pallets
    "Shuttle Storage (Tote Reserve)": 35000,  # goods-to-person tote shuttle

    "Replenishment Buffer": 2800,

    # Picking
    "Forward Pick Flat (Bins/Shelving)": 8400,
    "Forward Pick GOH (Hanging)": 4900,

    # Sort and pack spine
    "Pocket Sorter Buffer & Sequencing": 8400,
    "Packing & Consolidation": 11200,
    "Outbound Staging": 8400,
    "Shipping Dock": 3500,

    # Safety
    "HazMat/Segregated Storage": 700,
}

df_space = (
    pd.DataFrame({"Department": list(sqm_by_area.keys()),
                  "Planned Space (Sq m)": list(sqm_by_area.values())})
    .sort_values("Department")
    .reset_index(drop=True)
)

total_area = int(df_space["Planned Space (Sq m)"].sum())
assert total_area == 140_000, f"Total area is {total_area}, expected 140000"

# -----------------------------
# 2) Muther legend
# -----------------------------
df_legend = pd.DataFrame(
    {
        "Code": ["E", "A", "I", "O", "U", "X"],
        "Meaning": [
            "Especially important to be adjacent",
            "Absolutely necessary to be near",
            "Important to be near",
            "Ordinary closeness acceptable",
            "Unimportant",
            "Undesirable - keep apart",
        ],
        "Weight": [4, 3, 2, 1, 0, -4],
    }
)

# -----------------------------
# 3) Adjacency matrix (Muther SLP)
#    Start all U then set pairs
# -----------------------------
areas = list(sqm_by_area.keys())
adj = pd.DataFrame("U", index=areas, columns=areas)
for a in areas:
    adj.loc[a, a] = "-"

def set_pair(a: str, b: str, code: str) -> None:
    adj.loc[a, b] = code
    adj.loc[b, a] = code

# Inbound to putaway
set_pair("Inbound Dock", "Receiving/Staging", "E")
set_pair("Receiving/Staging", "QA & Inspection", "A")
set_pair("Receiving/Staging", "Cross-Dock", "A")
set_pair("Receiving/Staging", "Pallet Reserve Storage (Bulk)", "I")
set_pair("Receiving/Staging", "Shuttle Storage (Tote Reserve)", "I")

# Storage → replenishment → pick
set_pair("Pallet Reserve Storage (Bulk)", "Replenishment Buffer", "E")
set_pair("Shuttle Storage (Tote Reserve)", "Replenishment Buffer", "E")
set_pair("Replenishment Buffer", "Forward Pick Flat (Bins/Shelving)", "E")
set_pair("Replenishment Buffer", "Forward Pick GOH (Hanging)", "A")  # GOH restocks less often than flats

# Picking → sort/pack → ship
set_pair("Forward Pick Flat (Bins/Shelving)", "Pocket Sorter Buffer & Sequencing", "E")
set_pair("Forward Pick GOH (Hanging)", "Pocket Sorter Buffer & Sequencing", "I")
set_pair("Pocket Sorter Buffer & Sequencing", "Packing & Consolidation", "E")
set_pair("Forward Pick Flat (Bins/Shelving)", "Packing & Consolidation", "E")   # direct benches
set_pair("Forward Pick GOH (Hanging)", "Packing & Consolidation", "E")
set_pair("Packing & Consolidation", "Outbound Staging", "E")
set_pair("Outbound Staging", "Shipping Dock", "E")

# Reverse logistics
set_pair("Returns Intake & Triage", "QA & Inspection", "A")
set_pair("Returns Intake & Triage", "Returns Refurb & Repack", "I")
set_pair("Returns Refurb & Repack", "Packing & Consolidation", "I")

# Value add and cross-dock
set_pair("VAS/Kitting", "Packing & Consolidation", "I")
set_pair("Cross-Dock", "Outbound Staging", "A")
set_pair("Cross-Dock", "Shipping Dock", "A")

# Helpful but ordinary adjacencies
set_pair("Inbound Dock", "Shipping Dock", "O")  # door mgmt synergy
set_pair("VAS/Kitting", "Forward Pick Flat (Bins/Shelving)", "I")

# Safety separations
set_pair("HazMat/Segregated Storage", "Forward Pick Flat (Bins/Shelving)", "X")
set_pair("HazMat/Segregated Storage", "Forward Pick GOH (Hanging)", "X")
set_pair("HazMat/Segregated Storage", "Pocket Sorter Buffer & Sequencing", "X")
set_pair("HazMat/Segregated Storage", "Packing & Consolidation", "X")

# -----------------------------
# 4) Pair list with one-line justification
# -----------------------------
why_map = {
    frozenset(["Inbound Dock", "Receiving/Staging"]): "Direct unload into receiving minimizes dwell and touches.",
    frozenset(["Receiving/Staging", "QA & Inspection"]): "ID, sampling, and damages triaged immediately.",
    frozenset(["Receiving/Staging", "Cross-Dock"]): "Flow-through SKUs pivot quickly to outbound.",
    frozenset(["Pallet Reserve Storage (Bulk)", "Replenishment Buffer"]): "Bulk reserve feeds replenishment frequently.",
    frozenset(["Shuttle Storage (Tote Reserve)", "Replenishment Buffer"]): "GTP shuttle is the primary forward restock source.",
    frozenset(["Replenishment Buffer", "Forward Pick Flat (Bins/Shelving)"]): "High-frequency restocks need the shortest path.",
    frozenset(["Replenishment Buffer", "Forward Pick GOH (Hanging)"]): "GOH restocks are periodic; keep reasonably near.",
    frozenset(["Forward Pick Flat (Bins/Shelving)", "Pocket Sorter Buffer & Sequencing"]): "Pocket sorter buffers and sequences flats.",
    frozenset(["Forward Pick GOH (Hanging)", "Pocket Sorter Buffer & Sequencing"]): "Hanging flow interfaces as needed with pocket system.",
    frozenset(["Pocket Sorter Buffer & Sequencing", "Packing & Consolidation"]): "Sequenced items feed benches directly.",
    frozenset(["Forward Pick Flat (Bins/Shelving)", "Packing & Consolidation"]): "Direct bench feed for non-sequenced flats.",
    frozenset(["Forward Pick GOH (Hanging)", "Packing & Consolidation"]): "Hanging garments proceed to pack with minimal handling.",
    frozenset(["Packing & Consolidation", "Outbound Staging"]): "Packed orders queue immediately in lanes.",
    frozenset(["Outbound Staging", "Shipping Dock"]): "Short roll to doors protects carrier cutoffs.",
    frozenset(["Returns Intake & Triage", "QA & Inspection"]): "Triage and grading require QA diagnostics.",
    frozenset(["Returns Intake & Triage", "Returns Refurb & Repack"]): "Fast handoff to refurb minimizes rework dwell.",
    frozenset(["Returns Refurb & Repack", "Packing & Consolidation"]): "Repacked goods join outbound flow efficiently.",
    frozenset(["VAS/Kitting", "Packing & Consolidation"]): "Value-add completes near pack benches.",
    frozenset(["Cross-Dock", "Outbound Staging"]): "Door-to-lane flow for immediate departures.",
    frozenset(["Cross-Dock", "Shipping Dock"]): "Tight departures benefit from direct adjacency.",
    frozenset(["Inbound Dock", "Shipping Dock"]): "Door management and peak cross-staffing visibility.",
    frozenset(["HazMat/Segregated Storage", "Forward Pick Flat (Bins/Shelving)"]): "Keep hazardous SKUs away from people-dense areas.",
    frozenset(["HazMat/Segregated Storage", "Forward Pick GOH (Hanging)"]): "Separate flammables from GOH operations.",
    frozenset(["HazMat/Segregated Storage", "Pocket Sorter Buffer & Sequencing"]): "Avoid exposure near automated item buffers.",
    frozenset(["HazMat/Segregated Storage", "Packing & Consolidation"]): "No exposure at pack benches; ensure ventilation.",
    frozenset(["VAS/Kitting", "Forward Pick Flat (Bins/Shelving)"]): "Ticketing and relabeling consume each picks.",
}


rows_description = [
    ("Inbound Dock", "Truck arrival apron and doors for receipts; trailer check-in, seal verification, and unloading to receiving lanes."),
    ("Receiving/Staging", "ASN matching, counting, labeling, and exception handling; short-term buffer before QA or putaway."),
    ("QA & Inspection", "Identity checks, sampling, damage and compliance verification; routes good stock to storage or exceptions to returns."),
    ("Returns Intake & Triage", "Customer returns drop-off; visual check, basic testing, and disposition tagging for refurb, scrap, or restock."),
    ("Returns Refurb & Repack", "Light cleaning, steaming, retagging, and rebagging; prepares returnable items to re-enter stock or outbound flow."),
    ("Cross-Dock", "Flow-through for time-sensitive SKUs and vendor-prepped cartons; direct transfer from receiving to shipping lanes."),
    ("VAS/Kitting", "Ticketing, relabeling, bundling, and campaign prep; consumes each-pick items and feeds pack benches."),
    ("Pallet Reserve Storage (Bulk)", "Bulk and long-tail inventory on pallets; optimized for cube utilization and stable replenishment to forward areas."),
    ("Shuttle Storage (Tote Reserve)", "Automated tote buffer for goods-to-person; primary reserve for high-velocity each-pick assortment."),
    ("Replenishment Buffer", "Interface between reserves and forward pick; decouples restock waves and reduces aisle congestion."),
    ("Forward Pick Flat (Bins/Shelving)", "Each and case picks for flat-packed items; high-velocity slots and ergonomic pick faces close to pack."),
    ("Forward Pick GOH (Hanging)", "Each picks for garments on hanger; rails and drop points sized for sequence stability and presentation quality."),
    ("Pocket Sorter Buffer & Sequencing", "Dynamic item buffering and sequence building; feeds pack benches with correct order, size, and color runs."),
    ("Packing & Consolidation", "Order verification, dunnage, and labeling; consolidates lines from forward pick and pocket sorter for dispatch."),
    ("Outbound Staging", "Lane-based accumulation by carrier or route; protects late cutoffs and smooths trailer loading."),
    ("Shipping Dock", "Parcel and linehaul doors for dispatch; load planning, seal application, and handoff to carriers."),
    ("HazMat/Segregated Storage", "Controlled storage for regulated items like aerosols and perfumes; segregation and ventilation per policy."),
]

df_description = pd.DataFrame(rows_description, columns=["Department", "Description"])

adj.set_index(pd.Index(areas, name="From/To"), inplace=True)
rows = []
for i, a in enumerate(areas):
    for j in range(i+1, len(areas)):
        b = areas[j]
        code = str(adj.loc[a, b]).strip()
        if code in ("-", "U"):
            continue
        why = why_map.get(frozenset([a, b]))
        if why is None:
            why = "Operational coupling or safety policy dictates this rating."
        rows.append({"Area A": a, "Area B": b, "Code": code, "Why": why})

df_pairs = pd.DataFrame(rows).sort_values(["Code", "Area A", "Area B"]).reset_index(drop=True)

# -----------------------------
# 5) Write CSVs
# -----------------------------
outdir = Path("src/whlayout/data/instances/zalando")
(df_space.to_csv(outdir / "zalando_space_program_abs.csv", index=False))
(df_legend.to_csv(outdir / "muther_legend.csv", index=False))
(adj.loc[areas, areas].to_csv(outdir / "zalando_adjacency_matrix.csv", index=True))
(df_pairs.to_csv(outdir / "zalando_adjacency_pairs_why.csv", index=False))
(df_description.to_csv(outdir / "zalando_area_descriptions.csv", index=False))

print("Total site area:", total_area, "m²")
print("Wrote:")
for f in ["zalando_space_program_abs.csv", "muther_legend.csv", "zalando_adjacency_matrix.csv", "zalando_adjacency_pairs_why.csv"]:
    print(" -", f)

# whlayout

Warehouse Block Layout Optimization Models and Utilities

## Overview

`whlayout` is a Python package for modeling, solving, and visualizing warehouse block layout problems using Mixed-Integer Linear Programming (MILP) and Mixed-Integer Quadratically Constrained Programming (MIQCP) with Gurobi. It provides utilities for data handling, model construction, solution extraction, and plotting.

## Features

- Build flexible MILP/MIQCP models for warehouse block layout
- Support for department areas, adjacency/closeness ratings, clearances, anchors, and fixed departments
- Utilities for mapping adjacency ratings to weights
- Automatic building size suggestions
- Solution extraction and visualization (matplotlib, plotly)
- Example data and scripts for quick start

## Installation

1. Clone this repository:

   ```sh
   git clone <repo-url>
   cd wh-block-layout
   ```

2. Install dependencies (requires Python 3.8+ and Gurobi):

   ```sh
   pip install -r requirements.txt
   # or, for development
   pip install -e .
   ```

## Usage

### Example: MILP Block Layout

See `examples/example_MILP.py` for a full workflow. Key steps:

```python
from whlayout.model import build_block_layout_model
from whlayout.process import map_weights
from whlayout.layoututils import propose_building_sides
from whlayout.utils import extract_solution_table
from whlayout.plot import plot_layout
from whlayout.io import load_data

# Load data
areas = load_data("instances/17_depts/space_requirements.csv")
adjacency = load_data("instances/17_depts/adjacency_matrix.csv")

# Prepare model inputs
W = map_weights(adjacency, scheme="exp")
Bx, By = propose_building_sides(areas["Planned Space (sq m)"].to_dict())

# Build and solve model
model, var = build_block_layout_model(
    dept_list=areas.index.tolist(),
    areas_m2=areas["Planned Space (sq m)"].to_dict(),
    weights=W,
    building_x=Bx,
    building_y=By,
    # ... other options ...
)
model.optimize()

# Extract and plot solution
sol = extract_solution_table(areas.index.tolist(), Bx, By, var, areas_m2=areas["Planned Space (sq m)"].to_dict())
plot_layout(sol, Bx, By)
```

### Example: MIQCP Block Layout

See `examples/example_MIQCP.py` for a simpler use case with exact area constraints (bilinear) and custom clearances.

## Data

Sample data is provided in `src/whlayout/instances/17_depts/`:

- `space_requirements.csv`: Department areas
- `adjacency_matrix.csv`: Closeness ratings
- `adjacency_legend.csv`: Legend for ratings

## API Reference

- `build_block_layout_model`: Build and return a Gurobi model for block layout
- `map_weights`: Convert adjacency ratings to weights
- `propose_building_sides`: Suggest building dimensions
- `extract_solution_table`: Extract solution as a DataFrame
- `plot_layout`: Visualize the layout
- `load_data`: Load CSV data from package

## Requirements

- Python 3.8+
- Gurobi
- pandas
- matplotlib
- plotly
- PyQt6 (for interactive plots on VSCode)

## License

MIT License

## Author

Breno Alves Beirigo

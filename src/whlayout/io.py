import pandas as pd
import importlib.resources as pkg_resources
import whlayout.data

def load_data(filename: str) -> pd.DataFrame:
    """Load a CSV bundled with whlayout/data."""
    with pkg_resources.files(whlayout.data).joinpath(filename).open("rb") as f:
        return pd.read_csv(f, index_col=0)

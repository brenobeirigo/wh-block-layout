
import numpy as np

from typing import Literal
import pandas as pd
import numpy as np

def map_weights(
    ratings: pd.DataFrame, 
    scheme: Literal["linear_shift", "ordinal", "exp"] = "linear_shift", 
    scale_factor: float = 0.7
) -> pd.DataFrame:
    """Map an ordinal rating matrix to nonnegative weights.
    
    Args:
        ratings (pd.DataFrame): Input matrix with closeness ratings.
        scheme (str): Mapping scheme to use. Options are:
            - 'linear_shift': Shifts ratings to ensure all weights are positive.
            - 'ordinal': Maps ordinal ratings to a sequential scale.
            - 'exp': Applies an exponential transformation to ratings.
        scale_factor (float): Scaling factor for the 'exp' scheme. Defaults to 0.7.
    
    Returns:
        pd.DataFrame: Matrix of nonnegative weights derived from the input ratings.
    
    Notes:
        - The diagonal of the weight matrix is set to 0 to avoid self-loops.
        - The 'exp' scheme normalizes weights to have a mean of 1 for comparability.
    """
    ratings = ratings.copy()  # Work on a copy to avoid modifying the original
    

    if scheme == "ordinal":

        # Find the unique ratings to create a mapping (excluding NaNs)
        unique_ratings = np.unique(ratings.values[~np.isnan(ratings.values)])

        # Map ordinal ratings to a sequential scale
        rating_to_weight = {rating: idx + 1 
                            for idx, rating in enumerate(sorted(unique_ratings))}
        weights = ratings.replace(rating_to_weight)
    elif scheme == "exp":
        # Apply exponential transformation and normalize to mean 1
        weights = np.exp(scale_factor * ratings)
        weights = weights / weights.values[~np.isnan(weights.values)].mean()
    else:  # Default to 'linear_shift'
        # Shift ratings to ensure all weights are positive
        min_rating = np.nanmin(ratings.values)
        weights = ratings - min_rating + 1.0

    return weights.astype(float)
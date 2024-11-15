import pandas as pd
import geopandas as gpd
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError
from warnings import warn

def _check_df_load(df):
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return _check_gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")
    
def _check_gdf_load(gdf):
    if isinstance(gdf, str):
        if gdf.lower().endswith('csv'):
            return gpd.read_file(gdf, GEOM_POSSIBLE_NAMES="geometry",
                                 KEEP_GEOM_COLUMNS="NO")
        try:
            return gpd.read_file(gdf)
        except (DriverError, CPLE_OpenFailedError):
            warn(f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                 " path or it isn't a valid vector file. Returning an empty"
                 " GeoDataFrame.")
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")
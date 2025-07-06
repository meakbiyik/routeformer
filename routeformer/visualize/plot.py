"""Plotting functions for the GPS data."""
import io
import logging

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_gps_data_on_map(
    gps_df: pd.DataFrame,
    bounds_gdf=None,
    bounds=None,
    coordinate_system="EPSG:4326",
    figure_kwargs={"figsize": (10, 10), "frameon": False},
    plot_kwargs={"markersize": 50, "marker": "o", "color": "blue"},
    ax=None,
    offset=50,
    source=ctx.providers.OpenStreetMap.Mapnik,
) -> plt.Axes:
    """Plot the GPS data on a map.

    Parameters
    ----------
    gps_df : pd.DataFrame
        A dataframe with the GPS data. Must contain either the columns
        "x" and "y", or "latitude" and "longitude". If the former, the
        latitude is the y axis, longitude is the x axis in case the
        coordinate system is "EPSG:4326".
    bounds : list, optional
        The bounds of the map to plot, by default None.
    coordinate_system : str, optional
        The coordinate system of the GPS data, by default "EPSG:4326".
    figure_kwargs : dict, optional
        Keyword arguments for the Matplotlib figure, by default
        {"figsize": (10, 10), "frameon": False}.
    plot_kwargs : dict, optional
        Keyword arguments for the Matplotlib plot, by default
        {"markersize": 50, "marker": "o", "color": "blue"}.
    ax : plt.Axes, optional
        The Matplotlib axes object to plot on, by default None.
    source : ctx.providers, optional
        The Contextily map provider, by default ctx.providers.OpenStreetMap.Mapnik.

    Returns
    -------
    plt.Axes
        The Matplotlib axes object.
    """
    logger.info(f"Plotting GPS data with shape {gps_df.shape} on map.")
    if "x" in gps_df.columns and "y" in gps_df.columns:
        x, y = gps_df["x"].values, gps_df["y"].values
    elif "latitude" in gps_df.columns and "longitude" in gps_df.columns:
        x, y = gps_df["longitude"].values, gps_df["latitude"].values
    else:
        raise ValueError(
            "gps_df must contain either the columns 'x' and 'y', " "or 'latitude' and 'longitude'"
        )

    gdf = gpd.GeoDataFrame(
        gps_df,
        geometry=gpd.points_from_xy(x, y),
        crs=coordinate_system,
    )

    if bounds_gdf is not None:
        x_bounds, y_bounds = bounds_gdf["x"].values, bounds_gdf["y"].values
        gdf_bounds = gpd.GeoDataFrame(
            bounds_gdf,
            geometry=gpd.points_from_xy(x_bounds, y_bounds),
            crs=coordinate_system,
        )

    logger.debug(f"GPS data: {gdf.head()}")

    # Reproject the data to the Web Mercator projection (EPSG:3857)
    # gdf.to_crs("EPSG:3857", inplace=True)

    # Create a plot using Matplotlib
    if ax is None:
        fig = plt.figure(**figure_kwargs)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax.set_axis_off()

    if "color" in gdf.columns:
        plot_kwargs["color"] = "#00000000"
        plot_kwargs["edgecolor"] = gdf["color"].values

    gdf.plot(ax=ax, **plot_kwargs)

    logger.debug(f"Input bounds: {bounds}")
    if bounds is None:
        if bounds_gdf is not None:
            bounds = gdf_bounds.total_bounds
        else:
            bounds = gdf.total_bounds
    else:
        # convert bounds from GPS to "EPSG:3857"
        bounds = gpd.GeoSeries(
            gpd.points_from_xy(
                [bounds[1], bounds[3]], [bounds[0], bounds[2]], crs=coordinate_system
            ).to_crs("EPSG:3857")
        ).total_bounds
    logger.debug(f"Bounds: {bounds}")
    bounds = [
        _round_to_nearest_10(bounds[0] - offset),
        _round_to_nearest_10(bounds[1] - offset),
        _round_to_nearest_10(bounds[2] + offset),
        _round_to_nearest_10(bounds[3] + offset),
    ]
    logger.debug(f"Refined bounds: {bounds}")

    ax.set_xlim(
        [
            bounds[0],
            bounds[2],
        ]
    )
    ax.set_ylim(
        [
            bounds[1],
            bounds[3],
        ]
    )

    # Add a basemap using Contextily
    ctx.add_basemap(
        ax,
        source=source,
        zoom=19,
    )

    return ax


def render_figure_to_image(fig: plt.Figure, close=True) -> np.ndarray:
    """Render a Matplotlib figure to an image.

    Parameters
    ----------
    fig : plt.Figure
        The figure to render.

    Returns
    -------
    np.ndarray
        The image of the figure.
    """
    with io.BytesIO() as buff:
        plt.margins(0, 0)
        plt.tight_layout(pad=0)
        fig.savefig(buff, format="raw", pad_inches=0)
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    if close:
        fig.clear()
        plt.close(fig)

    return data.reshape((int(h), int(w), -1))


def _round_to_nearest_10(x):
    """Round a number to the nearest 10."""
    return int(np.ceil(x / 10.0)) * 10

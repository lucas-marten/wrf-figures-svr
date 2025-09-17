from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import numpy as np
from matplotlib.colors import ListedColormap


@dataclass(frozen=True)
class PlotConfig:
    plot_reflectivity: bool = True
    plot_mucape: bool = True
    plot_t2m: bool = True
    plot_vorticity: bool = True
    plot_total_precip: bool = True
    plot_wind_velocity: bool = True
    plot_precip_1h: bool = False
    plot_pw: bool = True
    plot_title: bool = True
    font_title: int = 12
    plot_type: str = "png"
    scale_contour: float = 1.0
    map_line: float = 0.5
    font_label: int = 15
    flip_barb: bool = True
    barb_spacing: Tuple[int, ...] = (10, 10, 10, 10, 10)
    map_color: str = "black"


@dataclass(frozen=True)
class ForecastMetadata:
    forecast_hour: int
    forecast_hour_str: str
    valid_datetime: dt.datetime
    initialization_title: str
    valid_title: str


@dataclass
class PlotData:
    lats: np.ndarray
    lons: np.ndarray
    refl_max: np.ndarray
    uh: np.ndarray
    mucape: np.ndarray
    mucin: np.ndarray
    ushear6km: np.ndarray
    vshear6km: np.ndarray
    T2m: np.ndarray
    u10m: np.ndarray
    v10m: np.ndarray
    u1km: np.ndarray
    v1km: np.ndarray
    vort1km: np.ndarray
    total_precip: np.ndarray
    wind10m_max: np.ndarray
    pw: np.ndarray
    T3km: np.ndarray
    u3km: np.ndarray
    v3km: np.ndarray
    precip_1h: np.ndarray | None = None
    lats_precip: np.ndarray | None = None
    lons_precip: np.ndarray | None = None


@dataclass
class PlotContext:
    config: PlotConfig
    datacrs: ccrs.CRS
    mapcrs: ccrs.CRS
    extent: Tuple[float, float, float, float]
    colormaps: Dict[str, ListedColormap]
    precip_colormap: ListedColormap
    precip_levels: Iterable[float]
    city_lons: np.ndarray
    city_lats: np.ndarray
    crepdecs_feature: ShapelyFeature | None = None


__all__ = [
    "PlotConfig",
    "ForecastMetadata",
    "PlotData",
    "PlotContext",
]

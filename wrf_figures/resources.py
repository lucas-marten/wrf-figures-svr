from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import cartopy.crs as ccrs
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature
from matplotlib.colors import ListedColormap

from .constants import COLORMAP_FILES


def load_listed_colormap(rgb_path: Path) -> ListedColormap:
    rgb_values = np.loadtxt(rgb_path, usecols=(0, 1, 2), unpack=True)
    colors = np.swapaxes(rgb_values, 0, 1) / 255.0
    return ListedColormap(colors)


def load_colormaps(colors_dir: Path) -> Dict[str, ListedColormap]:
    return {
        name: load_listed_colormap(colors_dir / filename)
        for name, filename in COLORMAP_FILES.items()
    }


def create_precip_colormap(json_path: Path) -> Tuple[ListedColormap, Iterable[float]]:
    with json_path.open() as json_file:
        colormap = json.load(json_file)
    original_data = colormap["data"]

    color_0mm = None
    filtered_data = []
    for level, color in original_data:
        if level == 0:
            color_0mm = color
        else:
            filtered_data.append([level, color])

    if color_0mm is None:
        raise ValueError("Colormap JSON missing zero level definition.")

    new_data = [[0, [255, 255, 255, 255]], [1, color_0mm]] + filtered_data
    levels = [entry[0] for entry in new_data]
    rgba_colors = np.array([entry[1] for entry in new_data], dtype=np.float32) / 255.0
    cmap_precip = ListedColormap(rgba_colors)
    return cmap_precip, levels


def load_crepdecs_feature(
    shapefile_path: Path, crs: ccrs.CRS | None = None
) -> ShapelyFeature | None:
    if not shapefile_path.exists():
        return None
    reader = shpreader.Reader(shapefile_path)
    return ShapelyFeature(
        reader.geometries(),
        crs or ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
    )


def get_city_coordinates(cities: Iterable[Tuple[str, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    lats = np.array([lat for _, lat, _ in cities])
    lons = np.array([lon for _, _, lon in cities])
    return lats, lons


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "load_listed_colormap",
    "load_colormaps",
    "create_precip_colormap",
    "load_crepdecs_feature",
    "get_city_coordinates",
    "ensure_directory",
]

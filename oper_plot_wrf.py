from __future__ import annotations

import datetime
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as mpatches
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import wrf


CIDADES_RS = [
    ("Porto Alegre", -30.0331, -51.2300),
    ("Caxias do Sul", -29.1678, -51.1794),
    ("Pelotas", -31.7719, -52.3426),
    ("Santa Maria", -29.6842, -53.8069),
    ("Gravataí", -29.9444, -50.9919),
    ("Viamão", -30.0819, -51.0233),
    ("Novo Hamburgo", -29.6783, -51.1309),
    ("Rio Grande", -32.0336, -52.0986),
    ("Alvorada", -29.9914, -51.0809),
    ("Passo Fundo", -28.2620, -52.4064),
    ("Sapucaia do Sul", -29.8276, -51.1440),
    ("Uruguaiana", -29.7596, -57.0853),
    ("Santa Cruz do Sul", -29.7184, -52.4250),
    ("Cachoeirinha", -29.9476, -51.0936),
    ("Bagé", -31.3297, -54.1069),
    ("Bento Gonçalves", -29.1662, -51.5165),
    ("Erechim", -27.6366, -52.2697),
    ("Guaíba", -30.1136, -51.3250),
    ("Lajeado", -29.4653, -51.9644),
    ("Santana do Livramento", -30.8900, -55.5328),
    ("Ijuí", -28.3880, -53.9194),
    ("Sapiranga", -29.6380, -51.0069),
    ("Vacaria", -28.5079, -50.9339),
    ("Cruz Alta", -28.6420, -53.6063),
    ("Alegrete", -29.7831, -55.7919),
    ("Venâncio Aires", -29.6143, -52.1932),
    ("Farroupilha", -29.2250, -51.3419),
    ("Cachoeira do Sul", -30.0397, -52.8939),
    ("Santa Rosa", -27.8689, -54.4800),
    ("Santo Ângelo", -28.2990, -54.2663),
    ("Carazinho", -28.2839, -52.7868),
    ("São Gabriel", -30.3341, -54.3219),
    ("Canguçu", -31.3928, -52.6753),
    ("Tramandaí", -29.9847, -50.1324),
    ("Capão da Canoa", -29.7644, -50.0281),
    ("Montenegro", -29.6822, -51.4678),
    ("Taquara", -29.6503, -50.7753),
    ("Parobé", -29.6261, -50.8344),
    ("Osório", -29.8860, -50.2700),
    ("Estância Velha", -29.6530, -51.1739),
    ("Campo Bom", -29.6747, -51.0619),
    ("Eldorado do Sul", -30.0847, -51.6197),
    ("Rosário do Sul", -30.2519, -54.9229),
    ("Dom Pedrito", -30.9828, -54.6717),
    ("Torres", -29.3339, -49.7339),
    ("Panambi", -28.2886, -53.5029),
    ("São Borja", -28.6600, -56.0047),
]

DIAS_SEMANA = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
FCSTHS = [f"{hour:02d}" for hour in range(85)]

COLORMAP_FILES = {
    "radar": "radar_py.rgb",
    "wind": "wind_py.rgb",
    "wind2": "wind2_py.rgb",
    "cape": "cape_py.rgb",
    "thetae": "thetae_py.rgb",
    "pw": "pw_py.rgb",
    "ir": "IR_py_m90_68.rgb",
}


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
    valid_datetime: datetime.datetime
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


def load_listed_colormap(rgb_path: Path) -> ListedColormap:
    rgb_values = np.loadtxt(rgb_path, usecols=(0, 1, 2), unpack=True)
    colors = np.swapaxes(rgb_values, 0, 1) / 255.0
    return ListedColormap(colors)


def load_colormaps(colors_dir: Path) -> Dict[str, ListedColormap]:
    return {name: load_listed_colormap(colors_dir / filename) for name, filename in COLORMAP_FILES.items()}


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


def load_crepdecs_feature(shapefile_path: Path) -> ShapelyFeature | None:
    if not shapefile_path.exists():
        return None
    reader = shpreader.Reader(shapefile_path)
    return ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor="black", linewidth=1.5)


def get_city_coordinates(cities: Iterable[Tuple[str, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    lats = np.array([lat for _, lat, _ in cities])
    lons = np.array([lon for _, _, lon in cities])
    return lats, lons


def build_forecast_metadata(init_datetime: datetime.datetime, forecast_hour_str: str) -> ForecastMetadata:
    forecast_hour = int(forecast_hour_str)
    valid_datetime = init_datetime + datetime.timedelta(hours=forecast_hour)
    dia_semana = DIAS_SEMANA[valid_datetime.weekday()]
    initialization_title = (
        f"WRF inicializado {init_datetime:%H}Z {init_datetime.day}/{init_datetime.month}/{init_datetime.year}"
    )
    valid_title = (
        f"Válido: {dia_semana} {valid_datetime:%Y%m%d} {valid_datetime:%H}Z\n"
        f"Horas de previsão: {forecast_hour_str}"
    )
    return ForecastMetadata(
        forecast_hour=forecast_hour,
        forecast_hour_str=forecast_hour_str,
        valid_datetime=valid_datetime,
        initialization_title=initialization_title,
        valid_title=valid_title,
    )


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_map_extent(lons: np.ndarray, lats: np.ndarray) -> Tuple[float, float, float, float]:
    minlon_map = float(np.min(lons) + 0.5)
    maxlon_map = float(np.max(lons) - 0.1)
    minlat_map = float(np.min(lats) + 0.15)
    maxlat_map = float(np.max(lats) - 0.16)
    return minlon_map, maxlon_map, minlat_map, maxlat_map


def collect_wrf_data(dataset: Dataset) -> PlotData:
    refl = wrf.getvar(dataset, "REFL_10CM")
    refl_max = np.max(refl, axis=0)
    lats, lons = wrf.latlon_coords(refl_max)

    uh = wrf.getvar(dataset, "updraft_helicity", timeidx=0, bottom=2000, top=5000)
    wind10m_max = wrf.getvar(dataset, "WSPD10MAX") * 3.6
    pw = wrf.getvar(dataset, "pw")
    T2m = wrf.getvar(dataset, "T2") - 273.15
    mucape, mucin, _, _ = wrf.getvar(dataset, "cape_2d")
    mucin = gaussian_filter(np.asarray(mucin), sigma=1.5)

    z = wrf.getvar(dataset, "height")
    terrain = wrf.getvar(dataset, "ter")
    z_agl = z - terrain

    u = wrf.getvar(dataset, "ua", units="m/s")
    v = wrf.getvar(dataset, "va", units="m/s")

    uvmet10 = wrf.getvar(dataset, "uvmet10")
    u10m = np.asarray(uvmet10[0])
    v10m = np.asarray(uvmet10[1])

    u1km = wrf.interplevel(u, z_agl, 1000.0).values
    v1km = wrf.interplevel(v, z_agl, 1000.0).values
    u3km = wrf.interplevel(u, z_agl, 3000.0).values
    v3km = wrf.interplevel(v, z_agl, 3000.0).values
    u6km = wrf.interplevel(u, z_agl, 6000.0).values
    v6km = wrf.interplevel(v, z_agl, 6000.0).values

    kt_factor = 1.94384
    ushear6km = (u6km - u10m) * kt_factor
    vshear6km = (v6km - v10m) * kt_factor
    u10m *= kt_factor
    v10m *= kt_factor
    u1km *= kt_factor
    v1km *= kt_factor
    u3km *= kt_factor
    v3km *= kt_factor

    T = wrf.getvar(dataset, "tc")
    T3km = gaussian_filter(wrf.interplevel(T, z_agl, 3000.0).values, sigma=1.5)

    absvort = wrf.getvar(dataset, "avo") / 100000.0
    mean_lat = float(np.mean(np.asarray(lats)))
    coriolis = 2 * 0.000072921 * np.sin(mean_lat)
    vort = absvort + coriolis
    vort1km = wrf.interplevel(vort, z_agl, 1000.0).values * 10000.0

    rainc = wrf.getvar(dataset, "RAINC")
    rainnc = wrf.getvar(dataset, "RAINNC")
    rainsh = wrf.getvar(dataset, "RAINSH")
    total_precip = np.asarray(rainc + rainnc + rainsh)

    return PlotData(
        lats=np.asarray(lats),
        lons=np.asarray(lons),
        refl_max=np.asarray(refl_max),
        uh=np.asarray(uh),
        mucape=np.asarray(mucape),
        mucin=mucin,
        ushear6km=ushear6km,
        vshear6km=vshear6km,
        T2m=np.asarray(T2m),
        u10m=u10m,
        v10m=v10m,
        u1km=u1km,
        v1km=v1km,
        vort1km=vort1km,
        total_precip=total_precip,
        wind10m_max=np.asarray(wind10m_max),
        pw=np.asarray(pw),
        T3km=T3km,
        u3km=u3km,
        v3km=v3km,
    )


def create_map_axes(context: PlotContext, include_states: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=context.datacrs)
    ax.set_extent(context.extent, context.mapcrs)
    add_base_features(ax, context, include_states=include_states)
    return fig, ax


def add_base_features(ax: plt.Axes, context: PlotContext, include_states: bool = True) -> None:
    cfg = context.config
    line_base = cfg.scale_contour * cfg.map_line
    map_color = cfg.map_color
    ax.add_feature(
        cfeature.LAND.with_scale("10m"),
        edgecolor=map_color,
        facecolor="white",
        linewidth=line_base + 1,
        zorder=1,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        edgecolor=map_color,
        linewidth=line_base + 1,
        zorder=2,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        edgecolor=map_color,
        linewidth=line_base + 1,
        zorder=2,
    )
    if include_states:
        ax.add_feature(
            cfeature.STATES.with_scale("10m"),
            edgecolor=map_color,
            linewidth=line_base,
            zorder=2,
        )


def add_cities(ax: plt.Axes, context: PlotContext) -> None:
    ax.scatter(
        context.city_lons,
        context.city_lats,
        s=15,
        marker="o",
        facecolors="black",
        edgecolors="white",
        transform=context.mapcrs,
        zorder=7,
        label="Cidades",
    )


def set_titles(ax: plt.Axes, config: PlotConfig, left_text: str, metadata: ForecastMetadata) -> None:
    if not config.plot_title:
        return
    ax.set_title(left_text, loc="left", fontsize=config.font_title)
    ax.set_title(metadata.valid_title, loc="right", fontsize=config.font_title + 1)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    trim_image(output_path)


def trim_image(image_path: Path) -> None:
    subprocess.run(["convert", str(image_path), "-trim", str(image_path)], check=False)


def log_plot_time(plot_number: str, start_time: float) -> None:
    duration = time.time() - start_time
    print(f"Tempo para PLOT {plot_number}: {duration:.2f} segundos")
    print()


def plot_reflectivity(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs = np.arange(5, 80, 5)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.refl_max,
        clevs,
        cmap=context.colormaps["radar"],
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal")
    cbar.ax.tick_params(labelsize=context.config.font_label)

    ax.contour(
        data.lons,
        data.lats,
        data.uh,
        levels=[-100, -50],
        colors=["black", "black"],
        linewidths=[2.5, 2.0],
        linestyles="solid",
        transform=context.mapcrs,
        zorder=3,
    )
    ax.contour(
        data.lons,
        data.lats,
        data.uh,
        levels=[30, 60],
        colors=["grey", "grey"],
        linewidths=[2.0, 2.5],
        linestyles="solid",
        transform=context.mapcrs,
        zorder=3,
    )

    legend_x, legend_y = 0.82, 0.06
    spacing_y = 0.04
    box_width, box_height = 0.16, 0.07

    background = mpatches.FancyBboxPatch(
        (legend_x - 0.015, legend_y - spacing_y - 0.015),
        box_width,
        box_height,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.7,
        zorder=2,
        boxstyle="round,pad=0.01",
    )
    ax.add_patch(background)

    ax.add_patch(
        mpatches.Circle(
            (legend_x, legend_y),
            0.015,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.add_patch(
        mpatches.Circle(
            (legend_x, legend_y),
            0.008,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="black",
            linewidth=2.5,
            zorder=3,
        )
    )
    ax.text(
        legend_x + 0.02,
        legend_y,
        "-50/-100 m²/s²",
        va="center",
        ha="left",
        transform=ax.transAxes,
        fontsize=context.config.font_label,
        zorder=3,
    )

    ax.add_patch(
        mpatches.Circle(
            (legend_x, legend_y - spacing_y),
            0.015,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="grey",
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.add_patch(
        mpatches.Circle(
            (legend_x, legend_y - spacing_y),
            0.008,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="grey",
            linewidth=2.5,
            zorder=3,
        )
    )
    ax.text(
        legend_x + 0.02,
        legend_y - spacing_y,
        "+30/+60 m²/s²",
        va="center",
        ha="left",
        transform=ax.transAxes,
        fontsize=context.config.font_label,
        zorder=3,
    )

    add_cities(ax, context)

    left_text = "Refletividade maxima na coluna (dBZ) e helicidade da corrente ascentente 2-5 km (m2/s2)\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_mucape(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs = np.arange(250, 5250, 250)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.mucape,
        clevs,
        cmap=context.colormaps["cape"],
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(
        shaded,
        ax=ax,
        ticks=[250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        pad=0.028,
        fraction=0.025,
        aspect=35,
        orientation="horizontal",
    )
    cbar.ax.tick_params(labelsize=context.config.font_label)

    barb_step = context.config.barb_spacing[0]
    ax.barbs(
        data.lons[::barb_step + 2, ::barb_step + 2],
        data.lats[::barb_step + 2, ::barb_step + 2],
        data.ushear6km[::barb_step + 2, ::barb_step + 2],
        data.vshear6km[::barb_step + 2, ::barb_step + 2],
        color="black",
        linewidth=context.config.scale_contour * 1.3,
        length=6.5,
        flip_barb=context.config.flip_barb,
        zorder=4,
    )

    cin_levels = [25, 50, 100, 150]
    cin_contour = ax.contour(
        data.lons,
        data.lats,
        data.mucin,
        cin_levels,
        colors=["green", "yellow", "red", "magenta"],
        linewidths=context.config.scale_contour * 2.0,
        transform=context.mapcrs,
        zorder=3,
    )
    ax.clabel(cin_contour, fmt="%d", fontsize=context.config.font_label)

    add_cities(ax, context)

    left_text = "CAPE da parcela mais instável (J/kg), cisalhamento 0-6-km\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_t2m(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs_T = np.arange(-40, 48, 2)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.T2m,
        clevs_T,
        cmap=context.colormaps["thetae"],
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(
        shaded,
        ax=ax,
        ticks=[-40, -30, -20, -10, 0, 10, 20, 30, 40],
        pad=0.028,
        fraction=0.025,
        aspect=35,
        orientation="horizontal",
    )
    cbar.ax.tick_params(labelsize=context.config.font_label)

    ax.contour(
        data.lons,
        data.lats,
        data.T2m,
        clevs_T,
        colors="grey",
        linestyles="-",
        linewidths=context.config.scale_contour * 0.8,
        transform=context.mapcrs,
        zorder=3,
    )
    labeled_contour = ax.contour(
        data.lons,
        data.lats,
        data.T2m,
        [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        colors="white",
        linestyles="-",
        linewidths=context.config.scale_contour * 1.5,
        transform=context.mapcrs,
        zorder=4,
    )
    ax.clabel(labeled_contour, fmt="%d", fontsize=context.config.font_label)

    barb_step = context.config.barb_spacing[0]
    ax.barbs(
        data.lons[::barb_step, ::barb_step],
        data.lats[::barb_step, ::barb_step],
        data.u10m[::barb_step, ::barb_step],
        data.v10m[::barb_step, ::barb_step],
        pivot="middle",
        color="black",
        linewidth=context.config.scale_contour * 1.3,
        length=6.5,
        flip_barb=context.config.flip_barb,
        zorder=6,
    )

    add_cities(ax, context)

    left_text = "Temperatura em 2m (C), vento em 10 m, pressão ao nível médio do mar (hPa)\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_vorticity(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs_vort = np.arange(-110, 0, 10)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.vort1km,
        clevs_vort,
        cmap=plt.cm.cividis,
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal")
    cbar.ax.tick_params(labelsize=context.config.font_label)

    ax.contour(
        data.lons,
        data.lats,
        data.refl_max,
        [50],
        colors="blue",
        linewidths=context.config.scale_contour * 1.5,
        transform=context.mapcrs,
        zorder=5,
    )

    barb_step = context.config.barb_spacing[0]
    ax.barbs(
        data.lons[::barb_step, ::barb_step],
        data.lats[::barb_step, ::barb_step],
        data.u1km[::barb_step, ::barb_step],
        data.v1km[::barb_step, ::barb_step],
        pivot="middle",
        color="black",
        linewidth=context.config.scale_contour * 1.3,
        length=6.5,
        flip_barb=context.config.flip_barb,
        zorder=6,
    )

    add_cities(ax, context)

    left_text = "Vorticidade (10^-4 s^-1) e vento em 1 km, refletividade 50 dBZ (azul)\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_total_precip(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context, include_states=False)

    norm_precip = BoundaryNorm(context.precip_levels, context.precip_colormap.N)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.total_precip,
        levels=context.precip_levels,
        cmap=context.precip_colormap,
        norm=norm_precip,
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal")
    cbar.ax.tick_params(labelsize=context.config.font_label)

    if context.crepdecs_feature is not None:
        ax.add_feature(context.crepdecs_feature, zorder=4)

    add_cities(ax, context)

    left_text = "Precipitação total (mm)\n" + metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_wind_velocity(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs_wind = np.arange(20, 120, 5)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.wind10m_max,
        levels=clevs_wind,
        cmap=context.colormaps["wind"],
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal")
    cbar.ax.tick_params(labelsize=context.config.font_label)

    ax.contour(
        data.lons,
        data.lats,
        data.refl_max,
        levels=[50],
        colors=["blue"],
        linewidths=[1.7],
        linestyles="solid",
        transform=context.mapcrs,
        zorder=3,
    )

    barb_step = context.config.barb_spacing[0]
    ax.barbs(
        data.lons[::barb_step, ::barb_step],
        data.lats[::barb_step, ::barb_step],
        data.u10m[::barb_step, ::barb_step],
        data.v10m[::barb_step, ::barb_step],
        pivot="middle",
        color="black",
        linewidth=context.config.scale_contour * 1.3,
        length=6.5,
        flip_barb=context.config.flip_barb,
        zorder=6,
    )

    add_cities(ax, context)

    left_text = "Velocidade do vento em 10 m máxima na última hora (km/h) e refletividade de 50 dBZ (azul)\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_precip_1h(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    if data.precip_1h is None or data.lats_precip is None or data.lons_precip is None:
        raise ValueError("Dados de precipitação 1h não disponíveis para plotagem.")

    fig, ax = create_map_axes(context, include_states=False)

    norm_precip = BoundaryNorm(context.precip_levels, context.precip_colormap.N)
    shaded = ax.contourf(
        data.lons_precip,
        data.lats_precip,
        data.precip_1h,
        levels=context.precip_levels,
        cmap=context.precip_colormap,
        norm=norm_precip,
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal")
    cbar.ax.tick_params(labelsize=context.config.font_label)

    if context.crepdecs_feature is not None:
        ax.add_feature(context.crepdecs_feature, zorder=4)

    add_cities(ax, context)

    left_text = "Precipitação na última hora (mm)\n" + metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_pw(data: PlotData, context: PlotContext, metadata: ForecastMetadata) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs_pw = np.arange(20, 72, 2)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.pw,
        clevs_pw,
        cmap=context.colormaps["pw"],
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(
        shaded,
        ax=ax,
        ticks=[20, 30, 40, 50, 60, 70],
        pad=0.028,
        fraction=0.025,
        aspect=35,
        orientation="horizontal",
    )
    cbar.ax.tick_params(labelsize=context.config.font_label)

    temp_contour = ax.contour(
        data.lons,
        data.lats,
        data.T3km,
        levels=np.arange(-30, 30, 2),
        colors=["red"],
        linewidths=[2.2],
        linestyles="--",
        transform=context.mapcrs,
        zorder=3,
    )
    ax.clabel(temp_contour, fmt="%d", fontsize=context.config.font_label)

    barb_step = context.config.barb_spacing[0]
    ax.barbs(
        data.lons[::barb_step, ::barb_step],
        data.lats[::barb_step, ::barb_step],
        data.u3km[::barb_step, ::barb_step],
        data.v3km[::barb_step, ::barb_step],
        pivot="middle",
        color="black",
        linewidth=context.config.scale_contour * 1.3,
        length=6.5,
        flip_barb=context.config.flip_barb,
        zorder=6,
    )

    add_cities(ax, context)

    left_text = "Água precipitável (mm), temperatura e vento em 3 km\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def main() -> None:
    start_time_total = time.time()

    config = PlotConfig()
    base_dir = Path(__file__).resolve().parent
    auxiliary_dir = base_dir / "arquivos_auxiliares"
    colors_dir = auxiliary_dir / "colors"

    colormaps = load_colormaps(colors_dir)
    precip_colormap, precip_levels = create_precip_colormap(auxiliary_dir / "colormap_qpe.json")
    crepdecs_feature = load_crepdecs_feature(auxiliary_dir / "crepdecs" / "CREPDECS_RS.shp")
    city_lats, city_lons = get_city_coordinates(CIDADES_RS)

    today = datetime.datetime.today()
    year = today.year
    julian_day = today.timetuple().tm_yday

    data_dir = Path(f"/storagefapesp/data/models/wrf/RAW/{year}/{julian_day:03d}")
    figures_dir = (base_dir / ".." / "figures" / f"{year}" / f"{julian_day:03d}").resolve()
    ensure_directory(figures_dir)

    target_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=julian_day - 1)
    init_datetime = datetime.datetime(target_date.year, target_date.month, target_date.day, 12)

    domain = "d02"
    datacrs = ccrs.PlateCarree()
    mapcrs = datacrs

    for fcsth in FCSTHS:
        metadata = build_forecast_metadata(init_datetime, fcsth)
        dataset_path = data_dir / f"wrfout_{domain}_{metadata.valid_datetime:%Y-%m-%d_%H}:00:00"
        if not dataset_path.exists():
            print(f"Arquivo {dataset_path} não encontrado. Pulando hora {fcsth}.")
            continue

        with Dataset(dataset_path) as dataset:
            data = collect_wrf_data(dataset)

        extent = compute_map_extent(data.lons, data.lats)
        context = PlotContext(
            config=config,
            datacrs=datacrs,
            mapcrs=mapcrs,
            extent=extent,
            colormaps=colormaps,
            precip_colormap=precip_colormap,
            precip_levels=precip_levels,
            city_lons=city_lons,
            city_lats=city_lats,
            crepdecs_feature=crepdecs_feature,
        )

        plot_name = f"{metadata.forecast_hour}.{config.plot_type}"

        print()
        print("________________________________________________________________________________")
        print(metadata.initialization_title)
        print(metadata.valid_title)
        print("________________________________________________________________________________")
        print()

        if config.plot_reflectivity:
            starttime = time.time()
            fig = plot_reflectivity(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "refl") / plot_name)
            log_plot_time("1", starttime)

        if config.plot_mucape:
            starttime = time.time()
            fig = plot_mucape(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "mucape") / plot_name)
            log_plot_time("2", starttime)

        if config.plot_t2m:
            starttime = time.time()
            fig = plot_t2m(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "t2m") / plot_name)
            log_plot_time("3", starttime)

        if config.plot_vorticity:
            starttime = time.time()
            fig = plot_vorticity(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "vort_1km") / plot_name)
            log_plot_time("4", starttime)

        if config.plot_total_precip:
            starttime = time.time()
            fig = plot_total_precip(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "total_precip") / plot_name)
            log_plot_time("5", starttime)

        if config.plot_wind_velocity:
            starttime = time.time()
            fig = plot_wind_velocity(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "wind_vel") / plot_name)
            log_plot_time("6", starttime)

        if config.plot_precip_1h:
            starttime = time.time()
            fig = plot_precip_1h(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "precip_1h") / plot_name)
            log_plot_time("7", starttime)

        if config.plot_pw:
            starttime = time.time()
            fig = plot_pw(data, context, metadata)
            save_figure(fig, ensure_directory(figures_dir / "pw_3km") / plot_name)
            log_plot_time("8", starttime)

    print(f"Tempo total: {time.time() - start_time_total:.2f} segundos")


if __name__ == "__main__":
    main()

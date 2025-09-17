from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import BoundaryNorm

from .models import ForecastMetadata, PlotConfig, PlotContext, PlotData


def create_map_axes(context: PlotContext, include_states: bool = True) -> tuple[plt.Figure, plt.Axes]:
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


def trim_image(image_path: Path) -> None:
    subprocess.run(["convert", str(image_path), "-trim", str(image_path)], check=False)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    trim_image(output_path)


def log_plot_time(plot_number: str, start_time: float) -> None:
    duration = time.time() - start_time
    print(f"Tempo para PLOT {plot_number}: {duration:.2f} segundos")
    print()


# Plotting functions below remain largely unchanged from the monolithic script.
# They are grouped here to keep the orchestration layer in cli.py lightweight.

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
    cbar = fig.colorbar(
        shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal"
    )
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
        data.lons[:: barb_step + 2, :: barb_step + 2],
        data.lats[:: barb_step + 2, :: barb_step + 2],
        data.ushear6km[:: barb_step + 2, :: barb_step + 2],
        data.vshear6km[:: barb_step + 2, :: barb_step + 2],
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
    cbar = fig.colorbar(
        shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal"
    )
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


def plot_total_precip(
    data: PlotData, context: PlotContext, metadata: ForecastMetadata
) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs_precip = [
        0.1,
        1,
        2,
        3,
        5,
        7,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        125,
        150,
        175,
        200,
    ]
    norm = BoundaryNorm(clevs_precip, context.colormaps["radar"].N)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.total_precip,
        clevs_precip,
        cmap=context.colormaps["radar"],
        norm=norm,
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(
        shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal"
    )
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

    add_cities(ax, context)

    left_text = "Precipitação acumulada (mm) e refletividade de 50 dBZ (azul)\n"
    left_text += metadata.initialization_title
    set_titles(ax, context.config, left_text, metadata)

    return fig


def plot_wind_velocity(
    data: PlotData, context: PlotContext, metadata: ForecastMetadata
) -> plt.Figure:
    fig, ax = create_map_axes(context)

    clevs_wind = np.arange(40, 125, 5)
    shaded = ax.contourf(
        data.lons,
        data.lats,
        data.wind10m_max,
        clevs_wind,
        cmap=context.colormaps["wind2"],
        transform=context.mapcrs,
        zorder=1,
    )
    cbar = fig.colorbar(
        shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal"
    )
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
    cbar = fig.colorbar(
        shaded, ax=ax, pad=0.028, fraction=0.025, aspect=35, orientation="horizontal"
    )
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


__all__ = [
    "create_map_axes",
    "add_base_features",
    "add_cities",
    "set_titles",
    "save_figure",
    "log_plot_time",
    "plot_reflectivity",
    "plot_mucape",
    "plot_t2m",
    "plot_vorticity",
    "plot_total_precip",
    "plot_wind_velocity",
    "plot_precip_1h",
    "plot_pw",
]

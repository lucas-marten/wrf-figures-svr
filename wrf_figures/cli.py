from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset

from .arguments import build_config_from_args, parse_arguments
from .constants import CIDADES_RS, FCSTHS, PROJECT_ROOT
from .data_processing import (
    build_forecast_metadata,
    collect_wrf_data,
    compute_map_extent,
)
from .models import PlotConfig, PlotContext
from .plotting import (
    log_plot_time,
    plot_mucape,
    plot_precip_1h,
    plot_pw,
    plot_reflectivity,
    plot_t2m,
    plot_total_precip,
    plot_vorticity,
    plot_wind_velocity,
    save_figure,
)
from .resources import (
    create_precip_colormap,
    ensure_directory,
    get_city_coordinates,
    load_colormaps,
    load_crepdecs_feature,
)


def resolve_raw_directory(args_raw_dir: str | None, target_date: dt.date) -> Path:
    if args_raw_dir:
        base_path = Path(args_raw_dir).expanduser()
    else:
        base_path = Path("/storagefapesp/data/models/wrf/RAW")
    return base_path / f"{target_date.year}" / f"{target_date.timetuple().tm_yday:03d}" / 00


def build_auxiliary_paths() -> tuple[Path, Path, Path]:
    auxiliary_dir = PROJECT_ROOT / "arquivos_auxiliares"
    colors_dir = auxiliary_dir / "colors"
    crepdecs_path = auxiliary_dir / "crepdecs" / "CREPDECS_RS.shp"
    return auxiliary_dir, colors_dir, crepdecs_path


def orchestrate_plots(
    raw_dir: Path,
    figures_dir: Path,
    config: PlotConfig,
    precip_levels: Iterable[float],
    precip_colormap: ListedColormap,
    colormaps: dict[str, ListedColormap],
    crepdecs_feature,
    city_lats,
    city_lons,
    init_datetime: dt.datetime,
    datacrs: ccrs.CRS,
    mapcrs: ccrs.CRS,
) -> None:
    domain = "d01"

    for fcsth in FCSTHS:
        metadata = build_forecast_metadata(init_datetime, fcsth)
        dataset_path = raw_dir / f"wrfout_{domain}_{metadata.valid_datetime:%Y-%m-%d_%H}:00:00"
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


def main(argv: Iterable[str] | None = None) -> None:
    start_time_total = time.time()

    args = parse_arguments(argv)
    config = build_config_from_args(args)

    target_date = args.rodada or dt.datetime.today().date()
    raw_dir = resolve_raw_directory(args.raw_dir, target_date)

    figures_dir = (
        PROJECT_ROOT
        / ".."
        / "figures"
        / f"{target_date.year}"
        / f"{target_date.timetuple().tm_yday:03d}"
    ).resolve()
    ensure_directory(figures_dir)

    _, colors_dir, crepdecs_path = build_auxiliary_paths()
    colormaps = load_colormaps(colors_dir)
    precip_colormap, precip_levels = create_precip_colormap(colors_dir / "colormap_qpe.json")
    crepdecs_feature = load_crepdecs_feature(crepdecs_path, ccrs.PlateCarree())
    city_lats, city_lons = get_city_coordinates(CIDADES_RS)

    init_datetime = dt.datetime(target_date.year, target_date.month, target_date.day, 12)

    datacrs = ccrs.PlateCarree()
    mapcrs = datacrs

    orchestrate_plots(
        raw_dir=raw_dir,
        figures_dir=figures_dir,
        config=config,
        precip_levels=precip_levels,
        precip_colormap=precip_colormap,
        colormaps=colormaps,
        crepdecs_feature=crepdecs_feature,
        city_lats=city_lats,
        city_lons=city_lons,
        init_datetime=init_datetime,
        datacrs=datacrs,
        mapcrs=mapcrs,
    )

    print(f"Tempo total: {time.time() - start_time_total:.2f} segundos")


__all__ = ["main"]

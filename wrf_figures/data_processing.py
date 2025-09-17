from __future__ import annotations

import datetime as dt
from typing import Tuple

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import wrf

from .constants import DIAS_SEMANA
from .models import ForecastMetadata, PlotData


def build_forecast_metadata(
    init_datetime: dt.datetime, forecast_hour_str: str
) -> ForecastMetadata:
    forecast_hour = int(forecast_hour_str)
    valid_datetime = init_datetime + dt.timedelta(hours=forecast_hour)
    dia_semana = DIAS_SEMANA[valid_datetime.weekday()]
    initialization_title = (
        f"WRF inicializado {init_datetime:%H}Z {init_datetime.day}/"
        f"{init_datetime.month}/{init_datetime.year}"
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


__all__ = [
    "build_forecast_metadata",
    "compute_map_extent",
    "collect_wrf_data",
]

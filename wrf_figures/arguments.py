from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
from typing import Iterable

from .constants import PLOT_ARGUMENTS
from .models import PlotConfig


def parse_run_date(value: str) -> dt.date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        "Data de rodada inválida. Utilize os formatos YYYYMMDD ou YYYY-MM-DD."
    )


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera figuras operacionais do WRF para diferentes variáveis meteorológicas."
        ),
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=sorted(PLOT_ARGUMENTS),
        metavar="PLOT",
        help=(
            "Lista de plots a serem gerados. Caso não seja informado, todas as opções "
            "serão processadas. Valores disponíveis: %(choices)s"
        ),
    )
    parser.add_argument(
        "--rodada",
        type=parse_run_date,
        help=(
            "Data da rodada do modelo no formato YYYYMMDD ou YYYY-MM-DD. "
            "Quando não informada, utiliza a data atual."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        help=(
            "Diretório base contendo os arquivos brutos do WRF. Quando não informado, "
            "utiliza a convenção padrão em /storagefapesp/data/models/wrf/RAW."
        ),
    )
    parser.add_argument(
        "--dominio",
        choices=("d01", "d02", "d03"),
        default="d01",
        help=(
            "Domínio do modelo WRF a ser utilizado (d01, d02 ou d03). Quando não informado, "
            "utiliza d01."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help=(
            "Diretório onde as figuras serão salvas. Quando não informado, utiliza o "
            "padrão figures/ano/dia_juliano relativo ao repositório."
        ),
    )
    return parser.parse_args(argv)


def build_config_from_args(
    args: argparse.Namespace, base_config: PlotConfig | None = None
) -> PlotConfig:
    config = base_config or PlotConfig()
    if args.plots is None:
        return config

    selected = set(args.plots)
    overrides = {
        attr_name: option in selected for option, attr_name in PLOT_ARGUMENTS.items()
    }
    return dataclasses.replace(config, **overrides)

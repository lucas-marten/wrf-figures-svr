from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

CIDADES_RS: Tuple[Tuple[str, float, float], ...] = (
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
)

DIAS_SEMANA: Tuple[str, ...] = ("Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom")

FCSTHS: Tuple[str, ...] = tuple(f"{hour:02d}" for hour in range(85))

COLORMAP_FILES: Dict[str, str] = {
    "radar": "radar_py.rgb",
    "wind": "wind_py.rgb",
    "wind2": "wind2_py.rgb",
    "cape": "cape_py.rgb",
    "thetae": "thetae_py.rgb",
    "pw": "pw_py.rgb",
    "ir": "IR_py_m90_68.rgb",
}

PLOT_ARGUMENTS: Dict[str, str] = {
    "reflectivity": "plot_reflectivity",
    "mucape": "plot_mucape",
    "t2m": "plot_t2m",
    "vorticity": "plot_vorticity",
    "total_precip": "plot_total_precip",
    "wind": "plot_wind_velocity",
    "precip_1h": "plot_precip_1h",
    "pw": "plot_pw",
}

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

__all__ = [
    "CIDADES_RS",
    "DIAS_SEMANA",
    "FCSTHS",
    "COLORMAP_FILES",
    "PLOT_ARGUMENTS",
    "BASE_DIR",
    "PROJECT_ROOT",
]

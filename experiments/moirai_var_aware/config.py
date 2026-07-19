from __future__ import annotations


HORIZONS = [1, 3, 5, 10, 21]
LOOKBACK = 60

SPLIT_INFO = {
    "DAX_40": {"train": 1983, "validation": 1104, "test": 972},
    "EuroNext_100": {"train": 2005, "validation": 1115, "test": 979},
    "IBEX_35": {"train": 1815, "validation": 1362, "test": 923},
    "KOSPI_index": {"train": 1944, "validation": 1109, "test": 881},
    "Nikkei_225": {"train": 1677, "validation": 1174, "test": 1062},
    "SMI": {"train": 1987, "validation": 1135, "test": 901},
    "snp500": {"train": 1988, "validation": 1109, "test": 927},
    "VN30_INDEX": {"train": 1971, "validation": 1096, "test": 925},
    "VN_INDEX": {"train": 1964, "validation": 1103, "test": 925},
}


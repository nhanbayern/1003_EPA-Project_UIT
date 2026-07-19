import torch

DEFAULT_SEQ_LEN = 60
DEFAULT_VOL_WINDOW = 60
HORIZONS = [1, 3, 5, 10, 21]
EVAL_INDICES = [0, 2, 4, 9, 20] # Indices corresponding to the horizons in a 21-length output (if applicable)
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed split counts per dataset (Train, Val, Test) - data range 2010-2025
FIXED_SPLITS = {
    "VN30_INDEX": (1971, 1096, 925),
    "VN_INDEX": (1964, 1103, 925),
    "DAX_40": (1983, 1104, 972),
    "EURONEXT_100": (2005, 1115, 979),
    "IBEX_35": (1815, 1362, 923),
    "KOSPI_INDEX": (1944, 1109, 881),
    "SMI": (1987, 1135, 901),
    "SNP500": (1988, 1109, 927),
    "NIKKEI_225": (1677, 1174, 1062),
}

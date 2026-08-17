# -*- coding: utf-8 -*-
"""
Direct ST-ResUNet-FPN Encoder-Decoder with Lead-Dependent Local Auxiliary Branch
=====================================================================================

Purpose
-------
A lightweight one-shot spatiotemporal encoder-decoder for tropical disturbance
pre-genesis position backtracking. Unlike autoregressive free-run models, this model
reads the known historical ERA5 sequence from TD formation time t back to t-72 h and
predicts all 24 historical center slots at once:

    step 1  -> t-3h
    step 2  -> t-6h
    ...
    step 24 -> t-72h

The 24 output slots are fixed as a maximum 72 h horizon. Storms with shorter official
pre-genesis DB records are handled by a valid_mask: unavailable target slots do not
contribute to loss or validation metrics.

Main design
-----------
Input per sequence:
    ERA5(t, t-3h, ..., t-72h), full-domain 100-180E, 0-40N, 0.25 deg grid
    shape = [T_in=25, C_aug, H=161, W=321]

C_aug = 9 ERA5 variables + TD anchor Gaussian map + latitude map + longitude map + lead-time map.

Output:
    heatmap_logits: [24, 1, 41, 81]
    offset_pred   : [24, 2, 41, 81]

where 41 x 81 is the stride-4 low-resolution grid. Offset is predicted in low-grid units.

Recommended usage
-----------------
python src/stage1/train_st_backnet.py --mode build_index
python src/stage1/train_st_backnet.py --mode train
python src/stage1/train_st_backnet.py --mode validate

or:
python src/stage1/train_st_backnet.py --mode all
"""

import os
import shutil
import json
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = Path(os.environ.get("TC_DATA_ROOT", REPOSITORY_ROOT / "data" / "raw"))
PUBLIC_OUTPUT_ROOT = Path(os.environ.get("TC_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))


# =========================================================
# 1. Config
# =========================================================

@dataclass
class Config:
    # ---------------- Fixed output root composition ----------------
    OUTPUT_ROOT: str = str(PUBLIC_OUTPUT_ROOT / "training" / "st_backnet")
    EXP_NAME: str = "st_backnet_lead_selected_local_canvas"
    # These fields are composed in __post_init__.
    OUT_DIR: str = field(init=False)
    INDEX_DIR: str = field(init=False)
    STATS_DIR: str = field(init=False)
    CKPT_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    VAL_DIR: str = field(init=False)
    MISSING_CHECK_DIR: str = field(init=False)
    SEQUENCE_INDEX_CSV: str = field(init=False)
    SEQUENCE_MISSING_DETAIL_CSV: str = field(init=False)
    STATS_JSON: str = field(init=False)
    BEST_CKPT: str = field(init=False)
    TRAIN_HISTORY_CSV: str = field(init=False)

    # ---------------- ERA5 paths ----------------
    VO850_DIR: str = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "vo850")
    RH700_DIR: str = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "rh700")
    TTR_DIR: str = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "ttr")
    VWS_DIR: str = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "vws200_850")
    UV925_DIR: str = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "uv925")
    MSLP_DIR: str = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "mslp")

    IBTRACS_PATH: str = str(RAW_DATA_ROOT / "ibtracs" / "western_north_pacific.csv")

    # ---------------- Native ERA5 grid definition ----------------
    LAT_START: float = 60.0
    LAT_END: float = -5.0
    LAT_STEP: float = -0.25
    LON_START: float = 80.0
    LON_STEP: float = 0.25
    MAX_COL: int = 401

    # ---------------- Full-domain model region ----------------
    DOMAIN_LAT_MIN: float = 0.0
    DOMAIN_LAT_MAX: float = 40.0
    DOMAIN_LON_MIN: float = 100.0
    DOMAIN_LON_MAX: float = 180.0
    DOMAIN_H: int = 161
    DOMAIN_W: int = 321
    LOW_STRIDE: int = 4
    LOW_H: int = 41
    LOW_W: int = 81

    # ---------------- IBTrACS columns ----------------
    COL_SID: str = "SID"
    COL_TIME: str = "ISO_TIME"
    COL_LAT: str = "USA_LAT"
    COL_LON: str = "USA_LON"
    COL_WIND: str = "USA_WIND"
    COL_SEASON: str = "SEASON"
    COL_STATUS: str = "USA_STATUS"

    # ---------------- Time and split ----------------
    TIME_STEP_HOURS: int = 3
    MAX_LEAD_HOURS: int = 72
    MAX_PRED_STEPS: int = 24       # 72 / 3
    INPUT_STEPS: int = 25          # t + 24 historical fields
    TRAIN_YEARS: Tuple[int, int] = (2000, 2017)
    VAL_YEARS: Tuple[int, int] = (2018, 2019)
    ALL_YEARS: Tuple[int, int] = (2000, 2019)

    # ---------------- Input variables ----------------
    USE_VO850_FILE_ALL_CHANNELS: bool = True
    USE_RH700: bool = True
    USE_TTR: bool = True
    USE_VWS: bool = True
    USE_UV925: bool = True
    USE_MSLP: bool = True

    # ---------------- Additional encodings ----------------
    USE_ANCHOR_HEATMAP: bool = True
    ANCHOR_HEATMAP_SIGMA_DEG: float = 1.0
    USE_LATLON_ENCODING: bool = True
    USE_LEAD_ENCODING: bool = True

    # ---------------- Label heatmap ----------------
    TARGET_HEATMAP_SIGMA_LOW_GRID: float = 1.0

    # ---------------- Training ----------------
    SEED: int = 42
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    EPOCHS: int = 50
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    AMP: bool = True
    GRAD_CLIP_NORM: float = 5.0
    BASE_CHANNELS: int = 32
    TEMPORAL_HIDDEN_CHANNELS: int = 128

    HEATMAP_LOSS_WEIGHT: float = 1.0
    OFFSET_LOSS_WEIGHT: float = 1.0
    COORD_LOSS_WEIGHT: float = 0.10
    SMOOTH_LOSS_WEIGHT: float = 0.03
    USE_LEAD_REWEIGHT: bool = True
    LEAD_WEIGHT_0_24: float = 1.0
    LEAD_WEIGHT_24_48: float = 1.2
    LEAD_WEIGHT_48_72: float = 1.5

    USE_EARLY_STOP: bool = True
    EARLY_STOP_PATIENCE: int = 8
    BEST_TIE_EPS: float = 1e-12

    # ---------------- Validation metrics ----------------
    PRIMARY_HIT_DEG: float = 0.25
    HIT_05_DEG: float = 0.50
    HIT_10_DEG: float = 1.00

    # ---------------- Anchor-distance prior decoding ----------------
    # This does not change the network. It only suppresses implausible far-field
    # heatmap peaks during decoding to reduce rare large outliers.
    USE_ANCHOR_DISTANCE_PRIOR_DECODE: bool = True
    PRIOR_MIN_SIGMA_DEG: float = 2.0
    PRIOR_SIGMA_PER_HOUR: float = 0.35
    PRIOR_POWER: float = 1.0
    PRIOR_HARD_RADIUS_DEG: float = 0.0  # 0 disables hard mask; use soft prior by default

    # ---------------- Local auxiliary branch ----------------
    # TD-anchor-centered local sequence used only as an auxiliary scale.
    # It should not replace the full-domain branch.
    #
    # Implementation note:
    #   The modified dataset uses a full-domain local canvas with one selected
    #   radius per input time step:
    #       0-24 h   : 10 deg
    #       24-48 h  : 15 deg
    #       48-72 h  : 20 deg
    #   Thus only one local canvas and one local mask are concatenated per time step.
    USE_LOCAL_AUX_BRANCH: bool = True
    LOCAL_RADIUS_DEG: float = 20.0  # fixed maximum crop radius; 20° -> 161x161 at 0.25°
    LOCAL_RADIUS_0_24_DEG: float = 10.0
    LOCAL_RADIUS_24_48_DEG: float = 15.0
    LOCAL_RADIUS_48_72_DEG: float = 20.0
    # Automatically recomputed from LOCAL_RADIUS_DEG in __post_init__/apply_args.
    LOCAL_H: int = 161
    LOCAL_W: int = 161
    LOCAL_LOW_H: int = 41
    LOCAL_LOW_W: int = 41
    LOCAL_LOSS_WEIGHT: float = 0.50
    LOCAL_FUSION_MAX_LEAD_HOURS: int = 72
    LOCAL_MIN_CONF: float = 0.02
    LOCAL_MIN_CONF_0_24: float = 0.02
    LOCAL_MIN_CONF_24_48: float = 0.04
    LOCAL_MIN_CONF_48_72: float = 0.08

    # ---------------- Data checking ----------------
    REQUIRE_FULL_ERA5_SEQUENCE: bool = True
    CHECK_NPY_READABLE: bool = True
    CHECK_NAN_INF: bool = True
    CHECK_SHAPE: bool = True
    VO850_MIN_CHANNELS: int = 3

    def __post_init__(self):
        self.OUT_DIR = os.path.join(self.OUTPUT_ROOT, self.EXP_NAME)
        self.INDEX_DIR = os.path.join(self.OUT_DIR, "index")
        self.STATS_DIR = os.path.join(self.OUT_DIR, "stats")
        self.CKPT_DIR = os.path.join(self.OUT_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.OUT_DIR, "logs")
        self.VAL_DIR = os.path.join(self.OUT_DIR, "validation")
        self.MISSING_CHECK_DIR = os.path.join(self.OUT_DIR, "missing_data_check_9var")
        self.SEQUENCE_INDEX_CSV = os.path.join(self.INDEX_DIR, "direct_st_sequence_index_train_val.csv")
        self.SEQUENCE_MISSING_DETAIL_CSV = os.path.join(self.MISSING_CHECK_DIR, "direct_st_sequence_missing_detail.csv")
        self.STATS_JSON = os.path.join(self.STATS_DIR, "direct_st_channel_stats_9var_full_domain.json")
        self.BEST_CKPT = os.path.join(self.CKPT_DIR, "best_direct_st_resunet_fpn_encoder_decoder_9var.pth")
        self.TRAIN_HISTORY_CSV = os.path.join(self.LOG_DIR, "training_history_direct_st_resunet_fpn_encoder_decoder_9var.csv")
        self._update_local_grid_sizes()

    def _conv_stride2_out_size(self, n: int) -> int:
        # SpatialResEncoder uses Conv2d(kernel=3, stride=2, padding=1).
        return int((int(n) + 1) // 2)

    def _update_local_grid_sizes(self) -> None:
        local_r = int(round(abs(float(self.LOCAL_RADIUS_DEG) / float(self.LON_STEP))))
        local_size = 2 * local_r + 1
        self.LOCAL_H = local_size
        self.LOCAL_W = local_size
        low1 = self._conv_stride2_out_size(local_size)
        low2 = self._conv_stride2_out_size(low1)
        self.LOCAL_LOW_H = low2
        self.LOCAL_LOW_W = low2


# =========================================================
# 2. Utilities
# =========================================================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def ensure_main_dirs(cfg: Config) -> None:
    for p in [cfg.OUT_DIR, cfg.INDEX_DIR, cfg.STATS_DIR, cfg.CKPT_DIR, cfg.LOG_DIR, cfg.VAL_DIR, cfg.MISSING_CHECK_DIR]:
        ensure_dir(p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def geo_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if not all(np.isfinite(v) for v in vals):
        return float("nan")
    return float(np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2))


def in_domain(cfg: Config, lat: float, lon: float) -> bool:
    return cfg.DOMAIN_LAT_MIN <= lat <= cfg.DOMAIN_LAT_MAX and cfg.DOMAIN_LON_MIN <= lon <= cfg.DOMAIN_LON_MAX


class GridSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lat_start = cfg.LAT_START
        self.lat_step = cfg.LAT_STEP
        self.lon_start = cfg.LON_START
        self.lon_step = cfg.LON_STEP
        self.max_row = int(round((cfg.LAT_END - cfg.LAT_START) / cfg.LAT_STEP)) + 1
        self.max_col = cfg.MAX_COL

        # Full-domain crop indices in native ERA5 grid.
        self.dom_row_north = int(round((cfg.DOMAIN_LAT_MAX - cfg.LAT_START) / cfg.LAT_STEP))
        self.dom_row_south = int(round((cfg.DOMAIN_LAT_MIN - cfg.LAT_START) / cfg.LAT_STEP))
        self.dom_col_west = int(round((cfg.DOMAIN_LON_MIN - cfg.LON_START) / cfg.LON_STEP))
        self.dom_col_east = int(round((cfg.DOMAIN_LON_MAX - cfg.LON_START) / cfg.LON_STEP))

        self.dom_rows = np.arange(self.dom_row_north, self.dom_row_south + 1)
        self.dom_cols = np.arange(self.dom_col_west, self.dom_col_east + 1)
        self.domain_h = len(self.dom_rows)
        self.domain_w = len(self.dom_cols)
        if self.domain_h != cfg.DOMAIN_H or self.domain_w != cfg.DOMAIN_W:
            raise ValueError(f"Domain size mismatch: got {self.domain_h}x{self.domain_w}, expected {cfg.DOMAIN_H}x{cfg.DOMAIN_W}")

    def geo_to_idx_float(self, lat: float, lon: float) -> Tuple[float, float]:
        row = (lat - self.lat_start) / self.lat_step
        col = (lon - self.lon_start) / self.lon_step
        return float(row), float(col)

    def idx_to_geo(self, row: float, col: float) -> Tuple[float, float]:
        lat = self.lat_start + row * self.lat_step
        lon = self.lon_start + col * self.lon_step
        return float(lat), float(lon)

    def geo_to_domain_yx(self, lat: float, lon: float) -> Tuple[float, float]:
        row, col = self.geo_to_idx_float(lat, lon)
        y = row - self.dom_row_north
        x = col - self.dom_col_west
        return float(y), float(x)

    def domain_yx_to_geo(self, y: float, x: float) -> Tuple[float, float]:
        row = self.dom_row_north + y
        col = self.dom_col_west + x
        return self.idx_to_geo(row, col)

    def domain_yx_to_low_yx(self, y: float, x: float) -> Tuple[float, float]:
        # Because 161->41 and 321->81: low index = high index / 4 exactly at boundaries.
        return float(y / self.cfg.LOW_STRIDE), float(x / self.cfg.LOW_STRIDE)

    def low_yx_to_domain_yx(self, ly: float, lx: float) -> Tuple[float, float]:
        return float(ly * self.cfg.LOW_STRIDE), float(lx * self.cfg.LOW_STRIDE)

    def crop_domain(self, arr_chw: np.ndarray) -> Optional[np.ndarray]:
        if arr_chw is None or arr_chw.ndim != 3:
            return None
        if arr_chw.shape[1] <= self.dom_row_south or arr_chw.shape[2] <= self.dom_col_east:
            return None
        return arr_chw[:, self.dom_row_north:self.dom_row_south + 1, self.dom_col_west:self.dom_col_east + 1].astype(np.float32)

    def make_latlon_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        # Normalized to [-1, 1]. Shape [H,W].
        lats = np.linspace(self.cfg.DOMAIN_LAT_MAX, self.cfg.DOMAIN_LAT_MIN, self.cfg.DOMAIN_H, dtype=np.float32)
        lons = np.linspace(self.cfg.DOMAIN_LON_MIN, self.cfg.DOMAIN_LON_MAX, self.cfg.DOMAIN_W, dtype=np.float32)
        lat_map = np.repeat(lats[:, None], self.cfg.DOMAIN_W, axis=1)
        lon_map = np.repeat(lons[None, :], self.cfg.DOMAIN_H, axis=0)
        lat_norm = 2.0 * (lat_map - self.cfg.DOMAIN_LAT_MIN) / (self.cfg.DOMAIN_LAT_MAX - self.cfg.DOMAIN_LAT_MIN) - 1.0
        lon_norm = 2.0 * (lon_map - self.cfg.DOMAIN_LON_MIN) / (self.cfg.DOMAIN_LON_MAX - self.cfg.DOMAIN_LON_MIN) - 1.0
        return lat_norm.astype(np.float32), lon_norm.astype(np.float32)


class NpyCache:
    def __init__(self, directory: str):
        self.directory = directory
        self.last_time: Optional[pd.Timestamp] = None
        self.last_data: Optional[np.ndarray] = None

    def load(self, time_obj: pd.Timestamp) -> Optional[np.ndarray]:
        time_obj = pd.to_datetime(time_obj)
        if self.last_time == time_obj and self.last_data is not None:
            return self.last_data
        if not self.directory or not os.path.isdir(self.directory):
            return None
        path = os.path.join(self.directory, time_obj.strftime("%Y%m%d%H.npy"))
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            if data.ndim not in (2, 3):
                return None
            self.last_time = time_obj
            self.last_data = data.astype(np.float32)
            return self.last_data
        except Exception:
            return None


class Era5Loader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vo850 = NpyCache(cfg.VO850_DIR)
        self.rh700 = NpyCache(cfg.RH700_DIR)
        self.ttr = NpyCache(cfg.TTR_DIR)
        self.vws = NpyCache(cfg.VWS_DIR)
        self.uv925 = NpyCache(cfg.UV925_DIR)
        self.mslp = NpyCache(cfg.MSLP_DIR)

    def channel_names(self) -> List[str]:
        names: List[str] = []
        if self.cfg.USE_VO850_FILE_ALL_CHANNELS:
            names += ["u850", "v850", "vo850"]
        if self.cfg.USE_RH700:
            names.append("rh700")
        if self.cfg.USE_TTR:
            names.append("ttr")
        if self.cfg.USE_VWS:
            names.append("vws200_850")
        if self.cfg.USE_UV925:
            names += ["u925", "v925"]
        if self.cfg.USE_MSLP:
            names.append("mslp")
        return names

    def get_full_fields(self, time_obj: pd.Timestamp) -> Optional[np.ndarray]:
        fields: List[np.ndarray] = []
        if self.cfg.USE_VO850_FILE_ALL_CHANNELS:
            data = self.vo850.load(time_obj)
            if data is None or data.ndim == 2 or data.shape[0] < 3:
                return None
            fields += [data[0], data[1], data[2]]
        if self.cfg.USE_RH700:
            data = self.rh700.load(time_obj)
            if data is None:
                return None
            fields.append(data[0] if data.ndim == 3 else data)
        if self.cfg.USE_TTR:
            data = self.ttr.load(time_obj)
            if data is None:
                return None
            fields.append(data[0] if data.ndim == 3 else data)
        if self.cfg.USE_VWS:
            data = self.vws.load(time_obj)
            if data is None:
                return None
            fields.append(data[0] if data.ndim == 3 else data)
        if self.cfg.USE_UV925:
            data = self.uv925.load(time_obj)
            if data is None or data.ndim != 3 or data.shape[0] < 2:
                return None
            fields += [data[0], data[1]]
        if self.cfg.USE_MSLP:
            data = self.mslp.load(time_obj)
            if data is None:
                return None
            fields.append(data[0] if data.ndim == 3 else data)
        if not fields:
            return None
        return np.stack(fields, axis=0).astype(np.float32)


# =========================================================
# 3. IBTrACS sequence index
# =========================================================

def load_ibtracs(cfg: Config) -> pd.DataFrame:
    try:
        df = pd.read_csv(cfg.IBTRACS_PATH, low_memory=False, skiprows=[1])
    except Exception:
        df = pd.read_csv(cfg.IBTRACS_PATH, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    df[cfg.COL_TIME] = pd.to_datetime(df[cfg.COL_TIME], format="mixed", errors="coerce")
    for c in [cfg.COL_LAT, cfg.COL_LON, cfg.COL_WIND, cfg.COL_SEASON]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[cfg.COL_SID, cfg.COL_TIME, cfg.COL_LAT, cfg.COL_LON, cfg.COL_SEASON])
    if cfg.COL_STATUS in df.columns and cfg.COL_WIND in df.columns:
        mask_fix = (
            df[cfg.COL_SEASON].between(2000, 2003)
            & (df[cfg.COL_STATUS].astype(str).str.upper() == "TD")
            & (df[cfg.COL_WIND] < 25)
        )
        df.loc[mask_fix, cfg.COL_STATUS] = "DB"
    return df


def determine_genesis_time(cfg: Config, group: pd.DataFrame) -> pd.Timestamp:
    status = group[cfg.COL_STATUS].astype(str).str.upper()
    cond_td = status.eq("TD")
    cond_other = (~status.isin(["DB", "TD"])) & (group[cfg.COL_WIND] >= 25)
    cand = group[cond_td | cond_other]
    return pd.to_datetime(cand[cfg.COL_TIME].min()) if not cand.empty else pd.to_datetime(group[cfg.COL_TIME].min())


def _find_genesis_row(cfg: Config, group: pd.DataFrame) -> Optional[pd.Series]:
    if group.empty:
        return None
    genesis_time = determine_genesis_time(cfg, group)
    cand = group[group[cfg.COL_TIME] == genesis_time]
    return cand.iloc[0] if not cand.empty else None


def _era5_file_exists_for_time(cfg: Config, t: pd.Timestamp) -> Tuple[bool, str]:
    fname = pd.to_datetime(t).strftime("%Y%m%d%H.npy")
    checks = []
    if cfg.USE_VO850_FILE_ALL_CHANNELS:
        checks.append(("vo850_pack", cfg.VO850_DIR))
    if cfg.USE_RH700:
        checks.append(("rh700", cfg.RH700_DIR))
    if cfg.USE_TTR:
        checks.append(("ttr", cfg.TTR_DIR))
    if cfg.USE_VWS:
        checks.append(("vws200_850", cfg.VWS_DIR))
    if cfg.USE_UV925:
        checks.append(("uv925_pack", cfg.UV925_DIR))
    if cfg.USE_MSLP:
        checks.append(("mslp", cfg.MSLP_DIR))
    for name, directory in checks:
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            return False, f"missing:{name}:{fname}"
        if cfg.CHECK_NPY_READABLE:
            try:
                arr = np.load(path)
                if cfg.CHECK_SHAPE:
                    if name == "vo850_pack" and (arr.ndim != 3 or arr.shape[0] < cfg.VO850_MIN_CHANNELS):
                        return False, f"bad_shape:{name}:{fname}:{arr.shape}"
                    if name == "uv925_pack" and (arr.ndim != 3 or arr.shape[0] < 2):
                        return False, f"bad_shape:{name}:{fname}:{arr.shape}"
                    if name not in ["vo850_pack", "uv925_pack"] and arr.ndim not in (2, 3):
                        return False, f"bad_shape:{name}:{fname}:{arr.shape}"
                if cfg.CHECK_NAN_INF:
                    if np.isinf(arr).any():
                        return False, f"inf:{name}:{fname}"
                    if np.isnan(arr).all():
                        return False, f"all_nan:{name}:{fname}"
            except Exception as e:
                return False, f"read_error:{name}:{fname}:{repr(e)}"
    return True, "ok"


def build_sequence_index(cfg: Config) -> pd.DataFrame:
    ensure_main_dirs(cfg)
    df = load_ibtracs(cfg)
    df = df[df[cfg.COL_SEASON].between(cfg.ALL_YEARS[0], cfg.ALL_YEARS[1])].copy()
    df = df.sort_values([cfg.COL_SID, cfg.COL_TIME]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    missing_records: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    def add_skip(k: str):
        skipped[k] = skipped.get(k, 0) + 1

    for sid, g in tqdm(df.groupby(cfg.COL_SID), desc="Building direct-ST sequence index"):
        g = g.sort_values(cfg.COL_TIME).reset_index(drop=True)
        if g.empty:
            continue
        genesis_row = _find_genesis_row(cfg, g)
        if genesis_row is None:
            add_skip("no_genesis_row")
            continue
        genesis_time = pd.to_datetime(genesis_row[cfg.COL_TIME])
        genesis_lat = float(genesis_row[cfg.COL_LAT])
        genesis_lon = float(genesis_row[cfg.COL_LON])
        season = int(genesis_row[cfg.COL_SEASON])
        name = genesis_row.get("NAME", "")

        if not in_domain(cfg, genesis_lat, genesis_lon):
            add_skip("genesis_out_of_domain")
            continue

        split = "ignore"
        if cfg.TRAIN_YEARS[0] <= season <= cfg.TRAIN_YEARS[1]:
            split = "train"
        elif cfg.VAL_YEARS[0] <= season <= cfg.VAL_YEARS[1]:
            split = "val"
        else:
            add_skip("outside_train_val_years")
            continue

        # Exact 3-hour official DB targets within 72 h before genesis.
        status = g[cfg.COL_STATUS].astype(str).str.upper()
        official_db = g[(g[cfg.COL_TIME] < genesis_time) & status.eq("DB")].copy()
        if official_db.empty:
            add_skip("no_pre_genesis_db")
            continue
        official_by_time = {pd.to_datetime(r[cfg.COL_TIME]): r for _, r in official_db.iterrows()}

        labels: List[Dict[str, Any]] = []
        for step in range(1, cfg.MAX_PRED_STEPS + 1):
            t = genesis_time - timedelta(hours=cfg.TIME_STEP_HOURS * step)
            r = official_by_time.get(t)
            if r is None:
                continue
            lat = float(r[cfg.COL_LAT])
            lon = float(r[cfg.COL_LON])
            if not in_domain(cfg, lat, lon):
                continue
            labels.append({
                "step": int(step),
                "lead_hour": float(step * cfg.TIME_STEP_HOURS),
                "time": pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S"),
                "lat": lat,
                "lon": lon,
                "status": str(r.get(cfg.COL_STATUS, "")).upper(),
                "wind": float(r.get(cfg.COL_WIND, np.nan)) if pd.notna(r.get(cfg.COL_WIND, np.nan)) else np.nan,
            })
        if len(labels) == 0:
            add_skip("no_valid_db_label_0_72")
            continue

        era5_ok = True
        reasons = []
        if cfg.REQUIRE_FULL_ERA5_SEQUENCE:
            for step_in in range(0, cfg.INPUT_STEPS):
                t = genesis_time - timedelta(hours=cfg.TIME_STEP_HOURS * step_in)
                ok, reason = _era5_file_exists_for_time(cfg, t)
                if not ok:
                    era5_ok = False
                    reasons.append(f"input_step_{step_in}:{reason}")
                    missing_records.append({
                        "SID": sid,
                        "SEASON": season,
                        "SPLIT": split,
                        "GENESIS_TIME": genesis_time,
                        "INPUT_STEP": step_in,
                        "TIME": t,
                        "REASON": reason,
                    })
        if not era5_ok:
            add_skip("missing_era5_sequence")
            continue

        rows.append({
            "SID": str(sid),
            "NAME": name,
            "SEASON": season,
            "SPLIT": split,
            "GENESIS_TIME": genesis_time,
            "GENESIS_LAT": genesis_lat,
            "GENESIS_LON": genesis_lon,
            "GENESIS_STATUS": str(genesis_row.get(cfg.COL_STATUS, "")).upper(),
            "GENESIS_WIND": genesis_row.get(cfg.COL_WIND, np.nan),
            "N_VALID_STEPS": len(labels),
            "MAX_LEAD_HOUR": max(x["lead_hour"] for x in labels),
            "LABELS_JSON": json.dumps(labels, ensure_ascii=False),
        })

    out = pd.DataFrame(rows)
    out.to_csv(cfg.SEQUENCE_INDEX_CSV, index=False)
    pd.DataFrame(missing_records).to_csv(cfg.SEQUENCE_MISSING_DETAIL_CSV, index=False)
    print(f"Saved sequence index: {cfg.SEQUENCE_INDEX_CSV}")
    print(f"Saved missing detail : {cfg.SEQUENCE_MISSING_DETAIL_CSV}")
    print(out["SPLIT"].value_counts(dropna=False) if not out.empty else "No sequences")
    print("Skipped:", skipped)
    return out


def read_sequence_index(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.SEQUENCE_INDEX_CSV):
        raise FileNotFoundError(f"Sequence index not found: {cfg.SEQUENCE_INDEX_CSV}. Run --mode build_index first.")
    index_df = pd.read_csv(cfg.SEQUENCE_INDEX_CSV, parse_dates=["GENESIS_TIME"])
    required = {"SEASON", "SPLIT", "GENESIS_TIME", "LABELS_JSON"}
    missing = sorted(required.difference(index_df.columns))
    if missing:
        raise RuntimeError(
            f"Cached sequence index is missing required columns {missing}; rebuild it with --force_rebuild_index."
        )

    seasons = pd.to_numeric(index_df["SEASON"], errors="coerce")
    expected_split = pd.Series("ignore", index=index_df.index, dtype="object")
    expected_split.loc[seasons.between(cfg.TRAIN_YEARS[0], cfg.TRAIN_YEARS[1])] = "train"
    expected_split.loc[seasons.between(cfg.VAL_YEARS[0], cfg.VAL_YEARS[1])] = "val"
    actual_split = index_df["SPLIT"].astype(str).str.lower()
    mismatch = actual_split.ne(expected_split)
    if bool(mismatch.any()):
        raise RuntimeError(
            "Cached sequence-index splits do not match the configured temporal split; "
            "rebuild it with --force_rebuild_index."
        )
    return index_df


# =========================================================
# 4. Data preprocessing and Dataset
# =========================================================

def make_gaussian_heatmap(size_hw: Tuple[int, int], y: float, x: float, sigma: float) -> np.ndarray:
    h, w = size_hw
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    heat = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def make_offset_target(size_hw: Tuple[int, int], y: float, x: float) -> np.ndarray:
    h, w = size_hw
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return np.stack([y - yy, x - xx], axis=0).astype(np.float32)


def normalize_domain_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x = np.where(np.isfinite(x), x, mean)
    x = (x - mean) / std
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_channel_stats(cfg: Config, index_df: pd.DataFrame, max_sequences: int = 300) -> Dict[str, Any]:
    ensure_main_dirs(cfg)
    grid = GridSystem(cfg)
    loader = Era5Loader(cfg)
    train_df = index_df[index_df["SPLIT"].astype(str).str.lower() == "train"].copy()
    if train_df.empty:
        raise RuntimeError("No train sequences for channel stats.")
    if len(train_df) > max_sequences:
        train_df = train_df.sample(max_sequences, random_state=cfg.SEED)
    sums = sqs = counts = None
    for _, r in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing full-domain channel stats"):
        genesis_time = pd.to_datetime(r["GENESIS_TIME"])
        for step_in in range(cfg.INPUT_STEPS):
            t = genesis_time - timedelta(hours=cfg.TIME_STEP_HOURS * step_in)
            fields = loader.get_full_fields(t)
            if fields is None:
                continue
            dom = grid.crop_domain(fields)
            if dom is None:
                continue
            if sums is None:
                c = dom.shape[0]
                sums = np.zeros(c, dtype=np.float64)
                sqs = np.zeros(c, dtype=np.float64)
                counts = np.zeros(c, dtype=np.float64)
            for ch in range(dom.shape[0]):
                vals = dom[ch]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                sums[ch] += vals.sum()
                sqs[ch] += np.square(vals).sum()
                counts[ch] += vals.size
    if sums is None or np.any(counts == 0):
        raise RuntimeError("Failed to compute channel stats; check ERA5 paths and sequence index.")
    mean = sums / counts
    var = np.maximum(sqs / counts - mean ** 2, 1e-12)
    std = np.sqrt(var)
    stats = {
        "channel_names": loader.channel_names(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "training_years": [int(cfg.TRAIN_YEARS[0]), int(cfg.TRAIN_YEARS[1])],
        "statistics_source": "training_split_only",
        "domain": {
            "lat_min": cfg.DOMAIN_LAT_MIN,
            "lat_max": cfg.DOMAIN_LAT_MAX,
            "lon_min": cfg.DOMAIN_LON_MIN,
            "lon_max": cfg.DOMAIN_LON_MAX,
            "H": cfg.DOMAIN_H,
            "W": cfg.DOMAIN_W,
        },
    }
    with open(cfg.STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved channel stats: {cfg.STATS_JSON}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def load_channel_stats(cfg: Config, index_df: pd.DataFrame) -> Dict[str, Any]:
    expected = Era5Loader(cfg).channel_names()
    expected_training_years = [int(cfg.TRAIN_YEARS[0]), int(cfg.TRAIN_YEARS[1])]
    if os.path.exists(cfg.STATS_JSON):
        with open(cfg.STATS_JSON, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if (
            stats.get("channel_names") == expected
            and stats.get("training_years") == expected_training_years
            and stats.get("statistics_source") == "training_split_only"
        ):
            return stats
        print("Existing channel stats lack matching training-split provenance; recomputing.")
    return compute_channel_stats(cfg, index_df)


def _lead_weight(cfg: Config, lead_hour: float) -> float:
    if not cfg.USE_LEAD_REWEIGHT:
        return 1.0
    if 0 < lead_hour <= 24:
        return cfg.LEAD_WEIGHT_0_24
    if 24 < lead_hour <= 48:
        return cfg.LEAD_WEIGHT_24_48
    if 48 < lead_hour <= 72:
        return cfg.LEAD_WEIGHT_48_72
    return 1.0


def _local_effective_radius_deg(cfg: Config, lead_hour: float) -> float:
    """Lead-dependent effective local radius in degrees.

    The local input crop itself is fixed at cfg.LOCAL_RADIUS_DEG, while this
    function controls which labels/predictions are considered valid for the
    local auxiliary branch at each lead time.
    """
    h = float(lead_hour)
    if 0 < h <= 24:
        return float(cfg.LOCAL_RADIUS_0_24_DEG)
    if 24 < h <= 48:
        return float(cfg.LOCAL_RADIUS_24_48_DEG)
    if 48 < h <= 72:
        return float(cfg.LOCAL_RADIUS_48_72_DEG)
    return float(cfg.LOCAL_RADIUS_DEG)


def _local_min_conf_by_lead(cfg: Config, lead_hour: float) -> float:
    h = float(lead_hour)
    if 0 < h <= 24:
        return float(getattr(cfg, "LOCAL_MIN_CONF_0_24", cfg.LOCAL_MIN_CONF))
    if 24 < h <= 48:
        return float(getattr(cfg, "LOCAL_MIN_CONF_24_48", cfg.LOCAL_MIN_CONF))
    if 48 < h <= 72:
        return float(getattr(cfg, "LOCAL_MIN_CONF_48_72", cfg.LOCAL_MIN_CONF))
    return float(cfg.LOCAL_MIN_CONF)


def _local_prediction_inside_effective_radius(cfg: Config, anchor_lat: float, anchor_lon: float,
                                              pred_lat: float, pred_lon: float, lead_hour: float) -> bool:
    if not all(np.isfinite(v) for v in [anchor_lat, anchor_lon, pred_lat, pred_lon, lead_hour]):
        return False
    return geo_distance_deg(anchor_lat, anchor_lon, pred_lat, pred_lon) <= _local_effective_radius_deg(cfg, lead_hour)


class BaseDirectSTSequenceDataset(Dataset):
    def __init__(self, cfg: Config, index_df: pd.DataFrame, split: str, stats: Dict[str, Any]):
        self.cfg = cfg
        self.grid = GridSystem(cfg)
        self.loader = Era5Loader(cfg)
        self.df = index_df[index_df["SPLIT"].astype(str).str.lower() == split].copy().reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No sequences for split={split}")
        self.stats = stats
        self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.lat_map, self.lon_map = self.grid.make_latlon_maps()

    def __len__(self):
        return len(self.df)

    def _make_input_sequence(self, genesis_time: pd.Timestamp, genesis_lat: float, genesis_lon: float) -> Optional[np.ndarray]:
        x_list: List[np.ndarray] = []
        anchor_y, anchor_x = self.grid.geo_to_domain_yx(genesis_lat, genesis_lon)
        sigma_grid = self.cfg.ANCHOR_HEATMAP_SIGMA_DEG / abs(self.cfg.LON_STEP)
        anchor_map = make_gaussian_heatmap((self.cfg.DOMAIN_H, self.cfg.DOMAIN_W), anchor_y, anchor_x, sigma=sigma_grid)[None]
        latlon_maps = np.stack([self.lat_map, self.lon_map], axis=0).astype(np.float32)
        for step_in in range(self.cfg.INPUT_STEPS):
            t = genesis_time - timedelta(hours=self.cfg.TIME_STEP_HOURS * step_in)
            fields = self.loader.get_full_fields(t)
            if fields is None:
                return None
            dom = self.grid.crop_domain(fields)
            if dom is None:
                return None
            dom = normalize_domain_with_stats(dom, self.mean, self.std)
            channels = [dom]
            if self.cfg.USE_ANCHOR_HEATMAP:
                channels.append(anchor_map)
            if self.cfg.USE_LATLON_ENCODING:
                channels.append(latlon_maps)
            if self.cfg.USE_LEAD_ENCODING:
                lead_norm = np.full((1, self.cfg.DOMAIN_H, self.cfg.DOMAIN_W),
                                    float(step_in * self.cfg.TIME_STEP_HOURS) / float(self.cfg.MAX_LEAD_HOURS),
                                    dtype=np.float32)
                channels.append(lead_norm)
            x_list.append(np.concatenate(channels, axis=0).astype(np.float32))
        return np.stack(x_list, axis=0).astype(np.float32)  # [T,C,H,W]

    def _make_targets(self, labels: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        heat = np.zeros((self.cfg.MAX_PRED_STEPS, 1, self.cfg.LOW_H, self.cfg.LOW_W), dtype=np.float32)
        offset = np.zeros((self.cfg.MAX_PRED_STEPS, 2, self.cfg.LOW_H, self.cfg.LOW_W), dtype=np.float32)
        offset_weight = np.zeros((self.cfg.MAX_PRED_STEPS, 1, self.cfg.LOW_H, self.cfg.LOW_W), dtype=np.float32)
        coord_low = np.zeros((self.cfg.MAX_PRED_STEPS, 2), dtype=np.float32)
        coord_geo = np.full((self.cfg.MAX_PRED_STEPS, 2), np.nan, dtype=np.float32)
        valid_mask = np.zeros((self.cfg.MAX_PRED_STEPS,), dtype=np.float32)
        lead_weight = np.ones((self.cfg.MAX_PRED_STEPS,), dtype=np.float32)
        for lab in labels:
            step = int(lab["step"])
            if not (1 <= step <= self.cfg.MAX_PRED_STEPS):
                continue
            k = step - 1
            lat, lon = float(lab["lat"]), float(lab["lon"])
            if not in_domain(self.cfg, lat, lon):
                continue
            y, x = self.grid.geo_to_domain_yx(lat, lon)
            ly, lx = self.grid.domain_yx_to_low_yx(y, x)
            if not (0 <= ly <= self.cfg.LOW_H - 1 and 0 <= lx <= self.cfg.LOW_W - 1):
                continue
            hm = make_gaussian_heatmap((self.cfg.LOW_H, self.cfg.LOW_W), ly, lx, self.cfg.TARGET_HEATMAP_SIGMA_LOW_GRID)
            off = make_offset_target((self.cfg.LOW_H, self.cfg.LOW_W), ly, lx)
            heat[k, 0] = hm
            offset[k] = off
            offset_weight[k, 0] = hm
            coord_low[k] = np.asarray([ly, lx], dtype=np.float32)
            coord_geo[k] = np.asarray([lat, lon], dtype=np.float32)
            valid_mask[k] = 1.0
            lead_weight[k] = _lead_weight(self.cfg, float(step * self.cfg.TIME_STEP_HOURS))
        return {
            "heatmap": heat,
            "offset": offset,
            "offset_weight": offset_weight,
            "coord_low": coord_low,
            "coord_geo": coord_geo,
            "valid_mask": valid_mask,
            "lead_weight": lead_weight,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        genesis_time = pd.to_datetime(r["GENESIS_TIME"])
        genesis_lat = float(r["GENESIS_LAT"])
        genesis_lon = float(r["GENESIS_LON"])
        labels = json.loads(r["LABELS_JSON"])
        x_seq = self._make_input_sequence(genesis_time, genesis_lat, genesis_lon)
        if x_seq is None:
            raise RuntimeError(f"Failed to build ERA5 input sequence for SID={r['SID']} time={genesis_time}")
        tgt = self._make_targets(labels)
        return {
            "x": torch.from_numpy(x_seq),
            "heatmap": torch.from_numpy(tgt["heatmap"]),
            "offset": torch.from_numpy(tgt["offset"]),
            "offset_weight": torch.from_numpy(tgt["offset_weight"]),
            "coord_low": torch.from_numpy(tgt["coord_low"]),
            "coord_geo": torch.from_numpy(tgt["coord_geo"]),
            "genesis_latlon": torch.tensor([genesis_lat, genesis_lon], dtype=torch.float32),
            "valid_mask": torch.from_numpy(tgt["valid_mask"]),
            "lead_weight": torch.from_numpy(tgt["lead_weight"]),
            "meta_index": torch.tensor(idx, dtype=torch.long),
        }


# =========================================================
# 5. Model: spatial encoder + ConvGRU temporal encoder + low-res decoder
# =========================================================

def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for g in [max_groups, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: Optional[int] = None, dilation: int = 1, act: bool = True):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(_valid_group_count(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, 3)
        self.conv2 = ConvGNAct(out_ch, out_ch, 3, act=False)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_valid_group_count(out_ch), out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        return self.act(y + self.shortcut(x))


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilations: Tuple[int, int, int] = (1, 2, 3)):
        super().__init__()
        branch_ch = max(out_ch // 4, 8)
        self.b0 = ConvGNAct(in_ch, branch_ch, 1, padding=0)
        self.b1 = ConvGNAct(in_ch, branch_ch, 3, dilation=dilations[0])
        self.b2 = ConvGNAct(in_ch, branch_ch, 3, dilation=dilations[1])
        self.b3 = ConvGNAct(in_ch, branch_ch, 3, dilation=dilations[2])
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, out_ch, 1, bias=False),
            nn.GroupNorm(_valid_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
            ResidualBlock(out_ch, out_ch, use_se=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1))


class SpatialResEncoder(nn.Module):
    """Shared 2D spatial encoder. Input 161x321 -> output 41x81."""
    def __init__(self, in_channels: int, base: int = 32, out_channels: int = 128):
        super().__init__()
        self.stem = ResidualBlock(in_channels, base, use_se=False)
        self.down1 = nn.Sequential(ConvGNAct(base, base * 2, 3, stride=2), ResidualBlock(base * 2, base * 2, use_se=True))
        self.down2 = nn.Sequential(ConvGNAct(base * 2, base * 4, 3, stride=2), ResidualBlock(base * 4, base * 4, use_se=True))
        self.aspp = ASPP(base * 4, base * 4)
        self.proj = nn.Sequential(
            ConvGNAct(base * 4, out_channels, 1, padding=0),
            ResidualBlock(out_channels, out_channels, use_se=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.aspp(x)
        x = self.proj(x)
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv_zr = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_dim, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h], dim=1)
        zr = torch.sigmoid(self.conv_zr(combined))
        z, r = torch.chunk(zr, 2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new


class DirectSTResUNetFPN(nn.Module):
    """One-shot multi-step full-domain backtracking model."""
    def __init__(self, in_channels: int, base: int = 32, hidden: int = 128, pred_steps: int = 24):
        super().__init__()
        self.pred_steps = pred_steps
        self.spatial_encoder = SpatialResEncoder(in_channels, base=base, out_channels=hidden)
        self.temporal_cell = ConvGRUCell(hidden, hidden, kernel_size=3)
        self.decoder = nn.Sequential(
            ResidualBlock(hidden, hidden, use_se=True),
            ResidualBlock(hidden, hidden, use_se=True),
        )
        self.heatmap_head = nn.Sequential(ConvGNAct(hidden, hidden // 2, 3), nn.Conv2d(hidden // 2, 1, 1))
        self.offset_head = nn.Sequential(ConvGNAct(hidden, hidden // 2, 3), nn.Conv2d(hidden // 2, 2, 1))

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_seq: [B,T,C,H,W]
        b, t, c, h, w = x_seq.shape
        x_flat = x_seq.reshape(b * t, c, h, w)
        feat_flat = self.spatial_encoder(x_flat)  # [B*T,D,41,81]
        _, d, lh, lw = feat_flat.shape
        feat = feat_flat.reshape(b, t, d, lh, lw)

        hidden_states = []
        h_state = None
        # Process from TD time to older times. Output target slot k uses hidden state at input index k+1.
        for i in range(t):
            h_state = self.temporal_cell(feat[:, i], h_state)
            hidden_states.append(h_state)
        target_states = hidden_states[1:self.pred_steps + 1]
        if len(target_states) < self.pred_steps:
            raise RuntimeError("Not enough input temporal states for pred_steps.")
        hs = torch.stack(target_states, dim=1)  # [B,24,D,Hl,Wl]
        hs_flat = hs.reshape(b * self.pred_steps, d, lh, lw)
        dec = self.decoder(hs_flat)
        heat = self.heatmap_head(dec).reshape(b, self.pred_steps, 1, lh, lw)
        offset = self.offset_head(dec).reshape(b, self.pred_steps, 2, lh, lw)
        return heat, offset


# =========================================================
# 6. Loss, decoding, metrics
# =========================================================

def soft_argmax_yx_sequence(heat_logits: torch.Tensor, offset_pred: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
    # heat_logits: [B,S,1,H,W], offset_pred: [B,S,2,H,W]
    b, s, _, h, w = heat_logits.shape
    logits = heat_logits.reshape(b, s, h * w) / max(float(temperature), 1e-6)
    prob = torch.softmax(logits, dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=heat_logits.device, dtype=torch.float32),
        torch.arange(w, device=heat_logits.device, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    y = torch.sum(prob * yy[None, None, :], dim=-1)
    x = torch.sum(prob * xx[None, None, :], dim=-1)
    if offset_pred is not None:
        off = offset_pred.permute(0, 1, 3, 4, 2).reshape(b, s, h * w, 2)
        y = y + torch.sum(prob * off[..., 0], dim=-1)
        x = x + torch.sum(prob * off[..., 1], dim=-1)
    return torch.stack([y, x], dim=-1)  # [B,S,2]


def masked_loss_fn(cfg: Config,
                   heat_logits: torch.Tensor,
                   offset_pred: torch.Tensor,
                   heat_target: torch.Tensor,
                   offset_target: torch.Tensor,
                   offset_weight: torch.Tensor,
                   coord_low: torch.Tensor,
                   valid_mask: torch.Tensor,
                   lead_weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # valid_mask/lead_weight: [B,S]
    step_weight = valid_mask * lead_weight
    denom = step_weight.sum().clamp_min(1e-6)

    heat_loss_map = F.binary_cross_entropy_with_logits(heat_logits, heat_target, reduction="none")  # [B,S,1,H,W]
    heat_loss_step = heat_loss_map.mean(dim=(2, 3, 4))
    heat_loss = (heat_loss_step * step_weight).sum() / denom

    off_diff = F.smooth_l1_loss(offset_pred, offset_target, reduction="none")  # [B,S,2,H,W]
    off_weight = offset_weight.expand_as(off_diff)
    off_step_num = (off_diff * off_weight).sum(dim=(2, 3, 4))
    off_step_den = off_weight.sum(dim=(2, 3, 4)).clamp_min(1e-6)
    off_loss_step = off_step_num / off_step_den
    off_loss = (off_loss_step * step_weight).sum() / denom

    pred_yx = soft_argmax_yx_sequence(heat_logits, offset_pred)
    coord_diff = F.smooth_l1_loss(pred_yx, coord_low, reduction="none").mean(dim=-1)
    coord_loss = (coord_diff * step_weight).sum() / denom

    smooth_loss = torch.tensor(0.0, device=heat_logits.device)
    if cfg.SMOOTH_LOSS_WEIGHT > 0:
        # Weak acceleration smoothness, only for consecutive valid target slots.
        if pred_yx.shape[1] >= 3:
            v1 = pred_yx[:, 1:-1] - pred_yx[:, :-2]
            v2 = pred_yx[:, 2:] - pred_yx[:, 1:-1]
            acc = torch.sqrt(torch.sum((v2 - v1) ** 2, dim=-1) + 1e-6)
            smask = valid_mask[:, :-2] * valid_mask[:, 1:-1] * valid_mask[:, 2:]
            sden = smask.sum().clamp_min(1e-6)
            smooth_loss = (acc * smask).sum() / sden

    total = (cfg.HEATMAP_LOSS_WEIGHT * heat_loss
             + cfg.OFFSET_LOSS_WEIGHT * off_loss
             + cfg.COORD_LOSS_WEIGHT * coord_loss
             + cfg.SMOOTH_LOSS_WEIGHT * smooth_loss)
    return total, {
        "HEAT_LOSS": heat_loss.detach(),
        "OFFSET_LOSS": off_loss.detach(),
        "COORD_LOSS": coord_loss.detach(),
        "SMOOTH_LOSS": smooth_loss.detach(),
    }


@torch.no_grad()
def decode_sequence_predictions(cfg: Config,
                                grid: GridSystem,
                                heat_logits: torch.Tensor,
                                offset_pred: torch.Tensor,
                                anchor_latlon: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
    """
    Decode sequence heatmaps to lat/lon centers.

    If cfg.USE_ANCHOR_DISTANCE_PRIOR_DECODE=True and anchor_latlon is provided,
    heatmap peaks are selected using:
        score = sigmoid(logit) * prior(distance_to_TD_anchor, lead_hour)^PRIOR_POWER
    This is a decoding-only modification; the network architecture and checkpoint
    format are unchanged.
    """
    prob = torch.sigmoid(heat_logits)
    b, s, _, h, w = prob.shape
    score = prob.clone()

    if getattr(cfg, "USE_ANCHOR_DISTANCE_PRIOR_DECODE", False) and anchor_latlon is not None:
        if not torch.is_tensor(anchor_latlon):
            anchor_latlon = torch.as_tensor(anchor_latlon, device=prob.device, dtype=prob.dtype)
        anchor_latlon = anchor_latlon.to(device=prob.device, dtype=prob.dtype)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=prob.device, dtype=prob.dtype),
            torch.arange(w, device=prob.device, dtype=prob.dtype),
            indexing="ij",
        )
        yy = yy.reshape(1, 1, 1, h, w)
        xx = xx.reshape(1, 1, 1, h, w)

        # Convert TD anchor lat/lon to low-resolution y/x coordinates.
        ay_list, ax_list = [], []
        for ib in range(b):
            alat = float(anchor_latlon[ib, 0].detach().cpu().item())
            alon = float(anchor_latlon[ib, 1].detach().cpu().item())
            dy, dx = grid.geo_to_domain_yx(alat, alon)
            ly, lx = grid.domain_yx_to_low_yx(dy, dx)
            ay_list.append(ly)
            ax_list.append(lx)
        ay = torch.tensor(ay_list, device=prob.device, dtype=prob.dtype).reshape(b, 1, 1, 1, 1)
        ax = torch.tensor(ax_list, device=prob.device, dtype=prob.dtype).reshape(b, 1, 1, 1, 1)

        lead_hours = torch.arange(1, s + 1, device=prob.device, dtype=prob.dtype).reshape(1, s, 1, 1, 1) * float(cfg.TIME_STEP_HOURS)
        sigma_deg = torch.clamp(float(cfg.PRIOR_SIGMA_PER_HOUR) * lead_hours, min=float(cfg.PRIOR_MIN_SIGMA_DEG))
        sigma_low = sigma_deg / (abs(float(cfg.LON_STEP)) * float(cfg.LOW_STRIDE))
        dist2 = (yy - ay) ** 2 + (xx - ax) ** 2
        prior = torch.exp(-dist2 / (2.0 * sigma_low ** 2 + 1e-6))
        if float(getattr(cfg, "PRIOR_HARD_RADIUS_DEG", 0.0)) > 0:
            hard_low = float(cfg.PRIOR_HARD_RADIUS_DEG) / (abs(float(cfg.LON_STEP)) * float(cfg.LOW_STRIDE))
            prior = torch.where(dist2 <= hard_low ** 2, prior, torch.zeros_like(prior))
        score = prob * torch.clamp(prior, min=1e-8).pow(float(cfg.PRIOR_POWER))

    flat = score.reshape(b, s, h * w)
    flat_idx = flat.argmax(dim=-1)
    # Report the original heatmap probability at selected position as confidence.
    prob_flat = prob.reshape(b, s, h * w)
    conf = torch.gather(prob_flat, dim=-1, index=flat_idx[..., None]).squeeze(-1)
    prior_score = flat.max(dim=-1).values
    py = (flat_idx // w).float()
    px = (flat_idx % w).float()
    off = offset_pred.permute(0, 1, 3, 4, 2).reshape(b, s, h * w, 2)
    gather_idx = flat_idx[..., None, None].expand(b, s, 1, 2)
    peak_off = torch.gather(off, dim=2, index=gather_idx).squeeze(2)
    pred_ly = py + peak_off[..., 0]
    pred_lx = px + peak_off[..., 1]
    pred_ly_np = pred_ly.detach().cpu().numpy()
    pred_lx_np = pred_lx.detach().cpu().numpy()
    lat = np.zeros_like(pred_ly_np, dtype=np.float32)
    lon = np.zeros_like(pred_lx_np, dtype=np.float32)
    for ib in range(b):
        for k in range(s):
            dy, dx = grid.low_yx_to_domain_yx(float(pred_ly_np[ib, k]), float(pred_lx_np[ib, k]))
            plat, plon = grid.domain_yx_to_geo(dy, dx)
            lat[ib, k] = plat
            lon[ib, k] = plon
    return {
        "pred_lat": lat.astype(np.float32),
        "pred_lon": lon.astype(np.float32),
        "conf": conf.detach().cpu().numpy().astype(np.float32),
        "prior_score": prior_score.detach().cpu().numpy().astype(np.float32),
        "pred_low_y": pred_ly_np.astype(np.float32),
        "pred_low_x": pred_lx_np.astype(np.float32),
    }

def summarize_errors(errors: np.ndarray, cfg: Config) -> Dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {
            "N": 0,
            "MEAN_ERROR_DEG": np.nan,
            "MEDIAN_ERROR_DEG": np.nan,
            "P75_ERROR_DEG": np.nan,
            "P90_ERROR_DEG": np.nan,
            "HIT_0P25_RATE": np.nan,
            "HIT_0P5_RATE": np.nan,
            "HIT_1P0_RATE": np.nan,
        }
    return {
        "N": int(len(errors)),
        "MEAN_ERROR_DEG": float(errors.mean()),
        "MEDIAN_ERROR_DEG": float(np.median(errors)),
        "P75_ERROR_DEG": float(np.quantile(errors, 0.75)),
        "P90_ERROR_DEG": float(np.quantile(errors, 0.90)),
        "HIT_0P25_RATE": float(np.mean(errors <= cfg.PRIMARY_HIT_DEG)),
        "HIT_0P5_RATE": float(np.mean(errors <= cfg.HIT_05_DEG)),
        "HIT_1P0_RATE": float(np.mean(errors <= cfg.HIT_10_DEG)),
    }


def _lead_bin_0_72(hours: float) -> str:
    h = float(hours)
    if 0 < h <= 24:
        return "0-24h"
    if 24 < h <= 48:
        return "24-48h"
    if 48 < h <= 72:
        return "48-72h"
    if h > 72:
        return ">72h"
    return "0h_or_negative"


def _summary_row(label_dict: Dict[str, Any], errors: np.ndarray, cfg: Config) -> Dict[str, Any]:
    row = dict(label_dict)
    row.update(summarize_errors(errors, cfg))
    return row


def build_step_and_bin_stats(point_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if point_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    db = point_df[point_df["POINT_TYPE"] == "db_pred"].copy()
    if db.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    db["ERROR_DEG"] = pd.to_numeric(db["ERROR_DEG"], errors="coerce")
    db["INFER_STEP"] = pd.to_numeric(db["INFER_STEP"], errors="coerce")
    db["HOURS_TO_GENESIS"] = pd.to_numeric(db["HOURS_TO_GENESIS"], errors="coerce")
    db = db[np.isfinite(db["ERROR_DEG"].values)].copy()

    step_rows: List[Dict[str, Any]] = []
    for step, g in db.groupby("INFER_STEP", sort=True):
        step_int = int(step)
        step_rows.append(_summary_row({"INFER_STEP": step_int, "LEAD_HOUR": step_int * cfg.TIME_STEP_HOURS}, g["ERROR_DEG"].values, cfg))
    step_df = pd.DataFrame(step_rows).sort_values("INFER_STEP").reset_index(drop=True) if step_rows else pd.DataFrame()

    db["LEAD_BIN"] = db["HOURS_TO_GENESIS"].apply(_lead_bin_0_72)
    bin_rows: List[Dict[str, Any]] = []
    for b in ["0-24h", "24-48h", "48-72h", ">72h"]:
        g = db[db["LEAD_BIN"] == b]
        if not g.empty:
            bin_rows.append(_summary_row({"LEAD_BIN": b}, g["ERROR_DEG"].values, cfg))
    bin_df = pd.DataFrame(bin_rows)

    final_rows: List[pd.Series] = []
    for _, g in db.groupby("SID", sort=False):
        g = g.sort_values("INFER_STEP")
        if not g.empty:
            final_rows.append(g.iloc[-1])
    final_raw = pd.DataFrame(final_rows)
    final_df = pd.DataFrame([_summary_row({"N_SEQUENCES": int(final_raw["SID"].nunique())}, final_raw["ERROR_DEG"].values, cfg)]) if not final_raw.empty else pd.DataFrame()
    return step_df, bin_df, final_df


def build_path_summary(point_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if point_df.empty:
        return pd.DataFrame()
    rows = []
    for sid, g in point_df.groupby("SID", sort=False):
        db = g[g["POINT_TYPE"] == "db_pred"].copy()
        db["ERROR_DEG"] = pd.to_numeric(db["ERROR_DEG"], errors="coerce") if not db.empty else np.nan
        valid = db[np.isfinite(db["ERROR_DEG"].values)].copy() if not db.empty else pd.DataFrame()
        first = g.iloc[0]
        official_steps_vals = pd.to_numeric(g.get("N_VALID_STEPS", pd.Series([np.nan])), errors="coerce").dropna()
        official_steps = int(official_steps_vals.iloc[0]) if len(official_steps_vals) else 0
        if not valid.empty:
            valid = valid.sort_values("INFER_STEP")
            last = valid.iloc[-1]
            s = summarize_errors(valid["ERROR_DEG"].values, cfg)
            final_err = float(last.get("ERROR_DEG", np.nan))
            max_hours = float(last.get("HOURS_TO_GENESIS", np.nan))
            final_conf = float(pd.to_numeric(last.get("CONF", np.nan), errors="coerce"))
        else:
            s = summarize_errors(np.asarray([], dtype=float), cfg)
            final_err = max_hours = final_conf = np.nan
        rows.append({
            "SID": sid,
            "NAME": first.get("NAME", ""),
            "SEASON": first.get("SEASON", np.nan),
            "SPLIT": first.get("SPLIT", ""),
            "GENESIS_TIME": first.get("GENESIS_TIME", pd.NaT),
            "N_VALID_STEPS": official_steps,
            "N_PREDICTED_DB_POINTS": int(len(valid)),
            "PATH_COMPLETE_0_72": bool(official_steps > 0 and len(valid) >= official_steps),
            "MAX_BACKTRACK_HOURS": max_hours,
            "FINAL_ERROR_DEG": final_err,
            "FINAL_CONF": final_conf,
            **{k: v for k, v in s.items() if k != "N"},
        })
    return pd.DataFrame(rows)


def _safe_filename_token(x: Any, max_len: int = 80) -> str:
    s = str(x) if x is not None else "NA"
    s = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s)
    s = s.strip("_") or "NA"
    return s[:max_len]



# =========================================================
# 7. Train and validation loops
# =========================================================

def make_dataloaders(cfg: Config, index_df: pd.DataFrame, stats: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_ds = DirectSTSequenceDataset(cfg, index_df, "train", stats)
    val_ds = DirectSTSequenceDataset(cfg, index_df, "val", stats)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)
    return train_loader, val_loader



def run_epoch(cfg: Config, model: nn.Module, loader: DataLoader, optimizer=None, scaler=None, device: str = "cuda", train: bool = False) -> Dict[str, Any]:
    grid = GridSystem(cfg)
    model.train() if train else model.eval()
    total_loss = total_heat = total_off = total_coord = total_smooth = 0.0
    n_batches = 0
    all_errors: List[float] = []
    all_valid = 0
    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        heat = batch["heatmap"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        offset_weight = batch["offset_weight"].to(device, non_blocking=True)
        coord_low = batch["coord_low"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        lead_weight = batch["lead_weight"].to(device, non_blocking=True)
        coord_geo = batch["coord_geo"]
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(cfg.AMP and device.startswith("cuda"))):
                heat_logits, offset_pred = model(x)
                loss, loss_parts = masked_loss_fn(cfg, heat_logits, offset_pred, heat, offset, offset_weight, coord_low, valid_mask, lead_weight)
            if train:
                if scaler is not None and cfg.AMP and device.startswith("cuda"):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if cfg.GRAD_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.GRAD_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                    optimizer.step()
        dec = decode_sequence_predictions(cfg, grid, heat_logits.detach(), offset_pred.detach(), batch.get("genesis_latlon", None))
        pred_lat = dec["pred_lat"]
        pred_lon = dec["pred_lon"]
        mask_np = valid_mask.detach().cpu().numpy()
        true_geo = coord_geo.numpy()
        for ib in range(mask_np.shape[0]):
            for k in range(mask_np.shape[1]):
                if mask_np[ib, k] > 0.5:
                    tlat, tlon = float(true_geo[ib, k, 0]), float(true_geo[ib, k, 1])
                    err = geo_distance_deg(float(pred_lat[ib, k]), float(pred_lon[ib, k]), tlat, tlon)
                    if np.isfinite(err):
                        all_errors.append(err)
        all_valid += int(mask_np.sum())
        total_loss += float(loss.detach().cpu())
        total_heat += float(loss_parts["HEAT_LOSS"].cpu())
        total_off += float(loss_parts["OFFSET_LOSS"].cpu())
        total_coord += float(loss_parts["COORD_LOSS"].cpu())
        total_smooth += float(loss_parts["SMOOTH_LOSS"].cpu())
        n_batches += 1
    summary = summarize_errors(np.asarray(all_errors), cfg)
    summary.update({
        "LOSS": total_loss / max(n_batches, 1),
        "HEAT_LOSS": total_heat / max(n_batches, 1),
        "OFFSET_LOSS": total_off / max(n_batches, 1),
        "COORD_LOSS": total_coord / max(n_batches, 1),
        "SMOOTH_LOSS": total_smooth / max(n_batches, 1),
        "N_VALID_TARGETS": int(all_valid),
    })
    return summary


def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer,
                    epoch: int,
                    cfg: Config,
                    stats: Dict[str, Any],
                    best_metric: float,
                    metric_info: Optional[Dict[str, Any]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(cfg),
        "stats": stats,
        # In this version, best_metric is the validation error used for top-k ranking.
        "best_metric": best_metric,
        "metric_info": metric_info or {},
    }, path)


def _topk_checkpoint_index_path(cfg: Config) -> str:
    return os.path.join(cfg.CKPT_DIR, "top5_hit10_hit05_checkpoints.json")


def _safe_float(x: Any, default: float = float("inf")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def _topk_record_sort_key(record: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
    """Rank checkpoints by higher Hit@1.0 and Hit@0.5.

    Primary metric   : validation HIT_1P0_RATE, descending.
    Tie-break metric : validation HIT_0P5_RATE, descending.
    Secondary tie    : validation mean error in degrees, ascending.
    Third tie        : validation median error in degrees, ascending.
    Final tie        : earlier epoch, ascending.
    """
    return (
        -_safe_float(record.get("val_hit_1p0_rate"), default=-float("inf")),
        -_safe_float(record.get("val_hit_0p5_rate"), default=-float("inf")),
        _safe_float(record.get("val_mean_error_deg")),
        _safe_float(record.get("val_median_error_deg")),
        int(record.get("epoch", 10 ** 9)),
    )


def _cleanup_old_topk_checkpoints(cfg: Config) -> None:
    """Start a new training run with a clean top-5 checkpoint list."""
    ensure_dir(cfg.CKPT_DIR)
    for name in os.listdir(cfg.CKPT_DIR):
        if name.startswith("checkpoint_epoch_") and name.endswith(".pth"):
            try:
                os.remove(os.path.join(cfg.CKPT_DIR, name))
            except OSError:
                pass
    for path in [_topk_checkpoint_index_path(cfg), cfg.BEST_CKPT]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _write_topk_checkpoint_index(cfg: Config, records: List[Dict[str, Any]]) -> None:
    ensure_dir(cfg.CKPT_DIR)
    payload = {
        "top_k": 5,
        "ranking_rule": "descending val_HIT_1P0_RATE; tie: descending val_HIT_0P5_RATE; tie: ascending val_MEAN_ERROR_DEG; tie: ascending val_MEDIAN_ERROR_DEG",
        "best_alias": cfg.BEST_CKPT,
        "checkpoints": records,
    }
    with open(_topk_checkpoint_index_path(cfg), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def update_topk_error_checkpoints(cfg: Config,
                                  model: nn.Module,
                                  optimizer,
                                  epoch: int,
                                  stats: Dict[str, Any],
                                  val_metrics: Dict[str, Any],
                                  topk_records: List[Dict[str, Any]],
                                  top_k: int = 5) -> Tuple[List[Dict[str, Any]], bool]:
    """Save only the top-k checkpoints with the highest validation Hit@1.0/Hit@0.5.

    Ranking rule:
        1) larger validation HIT_1P0_RATE is better;
        2) if tied, larger validation HIT_0P5_RATE is better;
        3) if still tied, smaller MEAN_ERROR_DEG is better;
        4) if still tied, smaller MEDIAN_ERROR_DEG is better.

    The function keeps at most five epoch-specific checkpoint files named
    checkpoint_epoch_*.pth. Validation metrics remain in the JSON index,
    not in checkpoint filenames. For compatibility with validation mode,
    cfg.BEST_CKPT is updated as a copy of the current rank-1 checkpoint.
    """
    val_hit10 = _safe_float(val_metrics.get("HIT_1P0_RATE"), default=float("nan"))
    val_hit05 = _safe_float(val_metrics.get("HIT_0P5_RATE"), default=float("nan"))
    val_mean = _safe_float(val_metrics.get("MEAN_ERROR_DEG"), default=float("nan"))
    val_median = _safe_float(val_metrics.get("MEDIAN_ERROR_DEG"), default=float("nan"))

    if not np.isfinite(val_hit10) or not np.isfinite(val_hit05):
        print("[TOPK-HIT] Skip checkpoint: validation HIT_1P0_RATE or HIT_0P5_RATE is not finite.")
        return topk_records, False

    filename = f"checkpoint_epoch_{epoch:03d}.pth"
    ckpt_path = os.path.join(cfg.CKPT_DIR, filename)
    candidate = {
        "rank": -1,
        "epoch": int(epoch),
        "path": ckpt_path,
        "filename": filename,
        "val_hit_1p0_rate": float(val_hit10),
        "val_hit_0p5_rate": float(val_hit05),
        "val_mean_error_deg": float(val_mean) if np.isfinite(val_mean) else float("nan"),
        "val_median_error_deg": float(val_median) if np.isfinite(val_median) else float("nan"),
        "val_loss": _safe_float(val_metrics.get("LOSS"), default=float("nan")),
        "val_n_valid_targets": int(val_metrics.get("N_VALID_TARGETS", 0)),
    }

    trial = sorted(topk_records + [candidate], key=_topk_record_sort_key)
    keep_paths = {r["path"] for r in trial[:top_k]}
    if ckpt_path not in keep_paths:
        worst = trial[top_k - 1] if len(trial) >= top_k else None
        if worst is not None:
            print(
                f"[TOPK-HIT] Not saved | epoch={epoch} "
                f"hit1.0={val_hit10:.4f} hit0.5={val_hit05:.4f}; "
                f"current top-{top_k} worst hit1.0={worst['val_hit_1p0_rate']:.4f} "
                f"hit0.5={worst['val_hit_0p5_rate']:.4f}"
            )
        return topk_records, False

    metric_info = {
        "ranking_metric": "val_HIT_1P0_RATE_then_val_HIT_0P5_RATE",
        "ranking_order": "descending",
        "tie_breakers": ["ascending val_MEAN_ERROR_DEG", "ascending val_MEDIAN_ERROR_DEG", "ascending epoch"],
        "val_metrics": val_metrics,
    }
    save_checkpoint(ckpt_path, model, optimizer, epoch, cfg, stats, best_metric=val_hit10, metric_info=metric_info)

    # Remove checkpoints falling out of top-k.
    new_records = trial[:top_k]
    removed_records = trial[top_k:]
    for rec in removed_records:
        path = rec.get("path", "")
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"[TOPK-HIT] Removed lower-ranked checkpoint: {path}")
            except OSError as e:
                print(f"[WARN] Failed to remove old checkpoint {path}: {repr(e)}")

    for rank, rec in enumerate(new_records, start=1):
        rec["rank"] = int(rank)

    # Keep cfg.BEST_CKPT as an alias to rank-1 for existing --mode validate behavior.
    if new_records:
        try:
            shutil.copy2(new_records[0]["path"], cfg.BEST_CKPT)
        except Exception as e:
            print(f"[WARN] Failed to update best checkpoint alias {cfg.BEST_CKPT}: {repr(e)}")

    _write_topk_checkpoint_index(cfg, new_records)

    current_is_best = bool(new_records and new_records[0]["path"] == ckpt_path)
    print("[TOPK-HIT] Current saved checkpoints:")
    for rec in new_records:
        print(
            f"  rank={rec['rank']} epoch={rec['epoch']} "
            f"hit1.0={rec['val_hit_1p0_rate']:.4f} "
            f"hit0.5={rec['val_hit_0p5_rate']:.4f} "
            f"val_mean={rec['val_mean_error_deg']:.6f} "
            f"median={rec['val_median_error_deg']:.6f} "
            f"file={rec['filename']}"
        )
    return new_records, current_is_best


def load_model_from_checkpoint(cfg: Config, checkpoint: str, device: str) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    stats = ckpt.get("stats", None)
    if stats is None:
        if os.path.exists(cfg.STATS_JSON):
            with open(cfg.STATS_JSON, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            raise RuntimeError("No stats found in checkpoint or STATS_JSON.")
    in_channels = get_augmented_in_channels(cfg, len(stats["channel_names"]))
    model = DirectSTResUNetFPN(in_channels=in_channels, base=cfg.BASE_CHANNELS,
                               hidden=cfg.TEMPORAL_HIDDEN_CHANNELS, pred_steps=cfg.MAX_PRED_STEPS).to(device)
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, stats, ckpt



@torch.no_grad()
def validate_model(cfg: Config, index_df: pd.DataFrame, checkpoint: str = "") -> None:
    ensure_main_dirs(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = checkpoint or cfg.BEST_CKPT
    print(f"Using device       : {device}")
    print(f"Checkpoint         : {checkpoint}")
    model, stats, _ = load_model_from_checkpoint(cfg, checkpoint, device)
    val_ds = DirectSTSequenceDataset(cfg, index_df, "val", stats)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)
    grid = GridSystem(cfg)
    records: List[Dict[str, Any]] = []
    model.eval()
    for batch in tqdm(val_loader, desc="Validate direct-ST paths"):
        x = batch["x"].to(device, non_blocking=True)
        heat_logits, offset_pred = model(x)
        dec = decode_sequence_predictions(cfg, grid, heat_logits, offset_pred, batch.get("genesis_latlon", None))
        meta_indices = batch["meta_index"].numpy().tolist()
        valid_mask = batch["valid_mask"].numpy()
        coord_geo = batch["coord_geo"].numpy()
        for ib, meta_idx in enumerate(meta_indices):
            r = val_ds.df.iloc[int(meta_idx)]
            genesis_time = pd.to_datetime(r["GENESIS_TIME"])
            records.append({
                "POINT_TYPE": "td_anchor",
                "SID": r["SID"],
                "NAME": r.get("NAME", ""),
                "SEASON": int(r["SEASON"]),
                "SPLIT": "val",
                "GENESIS_TIME": genesis_time,
                "TIME": genesis_time,
                "TARGET_TIME": genesis_time,
                "HOURS_TO_GENESIS": 0.0,
                "INFER_STEP": 0,
                "TRUE_LAT": float(r["GENESIS_LAT"]),
                "TRUE_LON": float(r["GENESIS_LON"]),
                "PRED_LAT": float(r["GENESIS_LAT"]),
                "PRED_LON": float(r["GENESIS_LON"]),
                "ERROR_DEG": 0.0,
                "CONF": np.nan,
                "HIT_0P25": True,
                "HIT_0P5": True,
                "HIT_1P0": True,
                "N_VALID_STEPS": int(r["N_VALID_STEPS"]),
            })
            for k in range(cfg.MAX_PRED_STEPS):
                if valid_mask[ib, k] <= 0.5:
                    continue
                step = k + 1
                target_time = genesis_time - timedelta(hours=cfg.TIME_STEP_HOURS * step)
                true_lat, true_lon = float(coord_geo[ib, k, 0]), float(coord_geo[ib, k, 1])
                pred_lat, pred_lon = float(dec["pred_lat"][ib, k]), float(dec["pred_lon"][ib, k])
                err = geo_distance_deg(pred_lat, pred_lon, true_lat, true_lon)
                conf = float(dec["conf"][ib, k])
                records.append({
                    "POINT_TYPE": "db_pred",
                    "SID": r["SID"],
                    "NAME": r.get("NAME", ""),
                    "SEASON": int(r["SEASON"]),
                    "SPLIT": "val",
                    "GENESIS_TIME": genesis_time,
                    "TIME": target_time,
                    "TARGET_TIME": target_time,
                    "HOURS_TO_GENESIS": float(step * cfg.TIME_STEP_HOURS),
                    "INFER_STEP": step,
                    "TRUE_LAT": true_lat,
                    "TRUE_LON": true_lon,
                    "PRED_LAT": pred_lat,
                    "PRED_LON": pred_lon,
                    "ERROR_DEG": err,
                    "CONF": conf,
                    "HIT_0P25": bool(np.isfinite(err) and err <= cfg.PRIMARY_HIT_DEG),
                    "HIT_0P5": bool(np.isfinite(err) and err <= cfg.HIT_05_DEG),
                    "HIT_1P0": bool(np.isfinite(err) and err <= cfg.HIT_10_DEG),
                    "N_VALID_STEPS": int(r["N_VALID_STEPS"]),
                })

    point_df = pd.DataFrame(records)
    point_csv = os.path.join(cfg.VAL_DIR, "val_path_points_direct_st_encoder_decoder.csv")
    summary_csv = os.path.join(cfg.VAL_DIR, "val_path_summary_direct_st_encoder_decoder.csv")
    step_csv = os.path.join(cfg.VAL_DIR, "val_path_step_stats_direct_st_encoder_decoder.csv")
    bin_csv = os.path.join(cfg.VAL_DIR, "val_path_lead_bin_stats_direct_st_encoder_decoder.csv")
    final_csv = os.path.join(cfg.VAL_DIR, "val_path_final_step_stats_direct_st_encoder_decoder.csv")
    point_df.to_csv(point_csv, index=False)
    summary_df = build_path_summary(point_df, cfg)
    summary_df.to_csv(summary_csv, index=False)
    step_df, bin_df, final_df = build_step_and_bin_stats(point_df, cfg)
    step_df.to_csv(step_csv, index=False)
    bin_df.to_csv(bin_csv, index=False)
    final_df.to_csv(final_csv, index=False)


    print(f"Saved val direct-ST path points : {point_csv}")
    print(f"Saved val direct-ST path summary: {summary_csv}")
    print(f"Saved val per-step stats       : {step_csv}")
    print(f"Saved val lead-bin stats       : {bin_csv}")
    print(f"Saved val final-step stats     : {final_csv}")

    if not point_df.empty:
        db_err = pd.to_numeric(point_df.loc[point_df["POINT_TYPE"] == "db_pred", "ERROR_DEG"], errors="coerce").values
        print("Val path-point summary [Direct ST Encoder-Decoder]:")
        print(summarize_errors(db_err, cfg))
    if not step_df.empty:
        cols = ["INFER_STEP", "LEAD_HOUR", "N", "MEAN_ERROR_DEG", "MEDIAN_ERROR_DEG", "HIT_0P5_RATE", "HIT_1P0_RATE"]
        print("\nVal per-step summary [Direct ST Encoder-Decoder] (step 1=t-3h, step 24=t-72h):")
        print(step_df[cols].to_string(index=False))
    if not bin_df.empty:
        cols = ["LEAD_BIN", "N", "MEAN_ERROR_DEG", "MEDIAN_ERROR_DEG", "HIT_0P5_RATE", "HIT_1P0_RATE"]
        print("\nVal lead-bin summary [Direct ST Encoder-Decoder]:")
        print(bin_df[cols].to_string(index=False))
    if not final_df.empty:
        print("\nVal final available step summary per sequence [Direct ST Encoder-Decoder]:")
        print(final_df.iloc[0].to_dict())


# =========================================================
# 8. CLI
# =========================================================




# =========================================================
# 8B. Variant A2: Lead-selected local canvas fusion
# =========================================================
# This variant uses a strict lead-dependent local-area input instead of
# concatenating all three local scales at every time step.
#
# For every input time step, only one TD-anchor-centered local region is embedded
# back into the full-domain canvas:
#     input_step 0-8   -> lead 0-24 h   -> ±10°
#     input_step 9-16  -> lead 27-48 h  -> ±15°
#     input_step 17-24 -> lead 51-72 h  -> ±20°
#
# Channel layout of the returned x is:
#     [global augmented input,
#      selected_local_canvas_by_lead,
#      selected_local_mask_by_lead]
#
# Therefore, with the default 9 ERA5 variables + 4 encodings:
#     C_aug = 13
#     C_in  = C_aug + C_aug + 1 = 27
#
# This keeps temporal length unchanged:
#     x = [T=25, C=27, H=161, W=321]
#
# If a local window exceeds the full-domain boundary, only the overlapping part is
# embedded into the global canvas. The out-of-domain part is clipped; the mask is
# 1 only at valid embedded positions and 0 elsewhere.

def _input_step_lead_hour(cfg: Config, input_step: int) -> float:
    """Return the lead hour of one input time step.

    input_step=0 corresponds to TD formation time t, so its lead is 0 h.
    input_step=1 corresponds to t-3 h, ..., input_step=24 corresponds to t-72 h.
    """
    return float(int(input_step) * int(cfg.TIME_STEP_HOURS))


def _local_radius_deg_by_lead(cfg: Config, lead_hour: float) -> float:
    """Strict lead-dependent local radius used by the input local canvas.

    The 0 h anchor frame is assigned to the 0-24 h group, because it is the known
    TD formation anchor and should use the nearest local context.
    """
    h = float(lead_hour)
    if 0.0 <= h <= 24.0:
        return float(cfg.LOCAL_RADIUS_0_24_DEG)
    if 24.0 < h <= 48.0:
        return float(cfg.LOCAL_RADIUS_24_48_DEG)
    if 48.0 < h <= 72.0:
        return float(cfg.LOCAL_RADIUS_48_72_DEG)
    return float(cfg.LOCAL_RADIUS_DEG)


def _lead_selected_local_radii_for_inputs(cfg: Config) -> List[float]:
    """Return the selected local radius for each input step."""
    return [
        _local_radius_deg_by_lead(cfg, _input_step_lead_hour(cfg, i))
        for i in range(int(cfg.INPUT_STEPS))
    ]


def _make_local_canvas_chw(arr_chw: np.ndarray, center_y: float, center_x: float,
                           radius_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return one anchor-centered local area embedded in full-domain canvas.

    arr_chw: [C,H,W]. Only the overlap between the ±radius local window and the
    global domain is copied. Out-of-domain area is clipped, not extrapolated.

    Returns
    -------
    canvas:
        [C,H,W], values outside the selected local window are zero.
    mask:
        [1,H,W], valid selected local window is 1, elsewhere 0.
    """
    c, h, w = arr_chw.shape
    canvas = np.zeros((c, h, w), dtype=np.float32)
    mask = np.zeros((1, h, w), dtype=np.float32)
    cy, cx = int(round(float(center_y))), int(round(float(center_x)))
    y1, y2 = cy - int(radius_grid), cy + int(radius_grid) + 1
    x1, x2 = cx - int(radius_grid), cx + int(radius_grid) + 1
    sy1, sy2 = max(0, y1), min(h, y2)
    sx1, sx2 = max(0, x1), min(w, x2)
    if sy1 < sy2 and sx1 < sx2:
        canvas[:, sy1:sy2, sx1:sx2] = arr_chw[:, sy1:sy2, sx1:sx2]
        mask[:, sy1:sy2, sx1:sx2] = 1.0
    return canvas.astype(np.float32), mask.astype(np.float32)


def _make_lead_selected_local_canvas_sequence(x_seq: np.ndarray, center_y: float, center_x: float,
                                              cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Build one lead-selected local canvas and mask for each input time step.

    Parameters
    ----------
    x_seq:
        [T,C,H,W] augmented full-domain sequence.
    center_y, center_x:
        TD anchor position in full-domain y/x coordinates.
    cfg:
        Config object.

    Returns
    -------
    local_canvas_seq:
        [T,C,H,W]. Each time step contains only the selected local radius for
        its own lead hour.
    local_mask_seq:
        [T,1,H,W]. One binary mask corresponding to local_canvas_seq.
    selected_radii_deg:
        List of selected radii for all T input steps.
    """
    selected_radii_deg = _lead_selected_local_radii_for_inputs(cfg)
    if len(selected_radii_deg) != x_seq.shape[0]:
        # This should not happen under normal config, but keeps the function safe
        # if INPUT_STEPS and x_seq length are modified separately.
        selected_radii_deg = [
            _local_radius_deg_by_lead(cfg, _input_step_lead_hour(cfg, i))
            for i in range(x_seq.shape[0])
        ]

    canvas_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    for input_step in range(x_seq.shape[0]):
        r_deg = float(selected_radii_deg[input_step])
        r_grid = int(round(abs(r_deg / float(cfg.LON_STEP))))
        canvas, mask = _make_local_canvas_chw(x_seq[input_step], center_y, center_x, r_grid)
        canvas_list.append(canvas)
        mask_list.append(mask)

    local_canvas_seq = np.stack(canvas_list, axis=0).astype(np.float32)  # [T,C,H,W]
    local_mask_seq = np.stack(mask_list, axis=0).astype(np.float32)      # [T,1,H,W]
    return local_canvas_seq, local_mask_seq, selected_radii_deg


class DirectSTSequenceDataset(BaseDirectSTSequenceDataset):
    """Dataset override: return x=[global, lead-selected local canvas, local mask]."""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        genesis_time = pd.to_datetime(r["GENESIS_TIME"])
        genesis_lat = float(r["GENESIS_LAT"])
        genesis_lon = float(r["GENESIS_LON"])
        labels = json.loads(r["LABELS_JSON"])

        x_seq = self._make_input_sequence(genesis_time, genesis_lat, genesis_lon)
        if x_seq is None:
            raise RuntimeError(f"Failed to build ERA5 input sequence for SID={r['SID']} time={genesis_time}")

        if getattr(self.cfg, "USE_LOCAL_AUX_BRANCH", True):
            ay, ax = self.grid.geo_to_domain_yx(genesis_lat, genesis_lon)
            # x_seq             : [T,C,H,W]
            # x_local_canvas    : [T,C,H,W], one selected radius per time step
            # x_local_mask      : [T,1,H,W]
            # x_fused           : [T,2*C+1,H,W]
            x_local_canvas, x_local_mask, _ = _make_lead_selected_local_canvas_sequence(x_seq, ay, ax, self.cfg)
            x_fused = np.concatenate([x_seq, x_local_canvas, x_local_mask], axis=1).astype(np.float32)
        else:
            x_fused = x_seq.astype(np.float32)

        tgt = self._make_targets(labels)
        return {
            "x": torch.from_numpy(x_fused),
            "heatmap": torch.from_numpy(tgt["heatmap"]),
            "offset": torch.from_numpy(tgt["offset"]),
            "offset_weight": torch.from_numpy(tgt["offset_weight"]),
            "coord_low": torch.from_numpy(tgt["coord_low"]),
            "coord_geo": torch.from_numpy(tgt["coord_geo"]),
            "genesis_latlon": torch.tensor([genesis_lat, genesis_lon], dtype=torch.float32),
            "valid_mask": torch.from_numpy(tgt["valid_mask"]),
            "lead_weight": torch.from_numpy(tgt["lead_weight"]),
            "meta_index": torch.tensor(idx, dtype=torch.long),
        }


def get_augmented_in_channels(cfg: Config, n_era5: int) -> int:
    """Return model input channels after lead-selected local fusion.

    Original augmented channels:
        C_aug = n_era5 + anchor + lat/lon + lead

    If local branch is enabled:
        [global C_aug] + [one selected local canvas C_aug] + [one local mask]
        C_in = 2*C_aug + 1

    Default:
        n_era5=9, extra=4 -> C_aug=13 -> C_in=27
    """
    extra = 0
    if cfg.USE_ANCHOR_HEATMAP:
        extra += 1
    if cfg.USE_LATLON_ENCODING:
        extra += 2
    if cfg.USE_LEAD_ENCODING:
        extra += 1

    c_aug = int(n_era5) + int(extra)
    if getattr(cfg, "USE_LOCAL_AUX_BRANCH", True):
        return 2 * c_aug + 1
    return c_aug


def _format_input_radius_schedule(cfg: Config) -> str:
    """Readable radius schedule for logging."""
    groups = [
        f"input_step 0-8 / lead 0-24h -> ±{float(cfg.LOCAL_RADIUS_0_24_DEG):.1f}°",
        f"input_step 9-16 / lead 27-48h -> ±{float(cfg.LOCAL_RADIUS_24_48_DEG):.1f}°",
        f"input_step 17-24 / lead 51-72h -> ±{float(cfg.LOCAL_RADIUS_48_72_DEG):.1f}°",
    ]
    return "; ".join(groups)


def train_model(cfg: Config, index_df: pd.DataFrame) -> None:
    set_seed(cfg.SEED)
    ensure_main_dirs(cfg)
    _cleanup_old_topk_checkpoints(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    stats = load_channel_stats(cfg, index_df)
    train_loader, val_loader = make_dataloaders(cfg, index_df, stats)
    in_channels = get_augmented_in_channels(cfg, len(stats["channel_names"]))
    print(f"Model variant: Lead-selected local canvas fusion")
    print(f"Input channels after fusion: {in_channels}")
    if getattr(cfg, "USE_LOCAL_AUX_BRANCH", True):
        print("Lead-selected local radius schedule: " + _format_input_radius_schedule(cfg))
        print("Only one local canvas and one mask are concatenated at each input time step")
    else:
        print("Local branch disabled; only global augmented input is used")

    model = DirectSTResUNetFPN(in_channels=in_channels, base=cfg.BASE_CHANNELS,
                               hidden=cfg.TEMPORAL_HIDDEN_CHANNELS, pred_steps=cfg.MAX_PRED_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and device.startswith("cuda")))

    # Save policy: keep only the 5 checkpoints with the highest validation Hit@1.0; tie by Hit@0.5.
    topk_records: List[Dict[str, Any]] = []
    best_hit10, best_hit05, no_improve = -float("inf"), -float("inf"), 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        tr = run_epoch(cfg, model, train_loader, optimizer=optimizer, scaler=scaler, device=device, train=True)
        va = run_epoch(cfg, model, val_loader, optimizer=None, scaler=None, device=device, train=False)
        scheduler.step()

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(cfg.TRAIN_HISTORY_CSV, index=False)
        print("Train:", tr)
        print("Val  :", va)

        topk_records, current_is_best = update_topk_error_checkpoints(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            stats=stats,
            val_metrics=va,
            topk_records=topk_records,
            top_k=5,
        )

        if current_is_best and topk_records:
            best_hit10 = float(topk_records[0]["val_hit_1p0_rate"])
            best_hit05 = float(topk_records[0]["val_hit_0p5_rate"])
            no_improve = 0
            print(
                f"[BEST-HIT] Updated top-1 | epoch={epoch} "
                f"hit1.0={best_hit10:.4f} hit0.5={best_hit05:.4f} "
                f"alias={cfg.BEST_CKPT}"
            )
        else:
            no_improve += 1

        if cfg.USE_EARLY_STOP and no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(
                f"[EARLY STOP] No validation Hit@1.0 / Hit@0.5 improvement for {no_improve} epochs. "
                f"Best hit1.0={best_hit10:.4f}, best hit0.5={best_hit05:.4f}"
            )
            break

    print(f"Training finished. Best validation Hit@1.0={best_hit10:.4f}, best Hit@0.5={best_hit05:.4f}")
    print(f"Top-5 checkpoint index: {_topk_checkpoint_index_path(cfg)}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="all", choices=["build_index", "train", "validate", "all"])
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--base_channels", type=int, default=None)
    p.add_argument("--hidden_channels", type=int, default=None)
    p.add_argument("--max_lead_hours", type=int, default=None)
    p.add_argument("--no_lead_reweight", action="store_true")
    p.add_argument("--no_anchor_prior_decode", action="store_true", help="Disable anchor-distance prior during decoding")
    p.add_argument("--prior_min_sigma_deg", type=float, default=None)
    p.add_argument("--prior_sigma_per_hour", type=float, default=None)
    p.add_argument("--prior_power", type=float, default=None)
    p.add_argument("--prior_hard_radius_deg", type=float, default=None)
    p.add_argument("--local_radius_deg", type=float, default=None, help="Fixed maximum local crop radius in degrees. Default 20.")
    p.add_argument("--local_radius_0_24_deg", type=float, default=None)
    p.add_argument("--local_radius_24_48_deg", type=float, default=None)
    p.add_argument("--local_radius_48_72_deg", type=float, default=None)
    p.add_argument("--local_loss_weight", type=float, default=None)
    p.add_argument("--local_fusion_max_lead_hours", type=int, default=None)
    p.add_argument("--local_min_conf", type=float, default=None)
    p.add_argument("--local_min_conf_0_24", type=float, default=None)
    p.add_argument("--local_min_conf_24_48", type=float, default=None)
    p.add_argument("--local_min_conf_48_72", type=float, default=None)
    p.add_argument("--smooth_loss_weight", type=float, default=None)
    p.add_argument("--coord_loss_weight", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--force_rebuild_index", action="store_true")
    p.add_argument("--no_rh700", action="store_true")
    p.add_argument("--no_ttr", action="store_true")
    p.add_argument("--no_vws", action="store_true")
    p.add_argument("--no_uv925", action="store_true")
    p.add_argument("--no_mslp", action="store_true")
    return p.parse_args()


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    # Recompose paths if output root/name is changed.
    if args.output_root is not None:
        cfg.OUTPUT_ROOT = str(args.output_root)
    if args.exp_name is not None:
        cfg.EXP_NAME = str(args.exp_name)
    cfg.__post_init__()

    if args.batch_size is not None:
        cfg.BATCH_SIZE = int(args.batch_size)
    if args.epochs is not None:
        cfg.EPOCHS = int(args.epochs)
    if args.lr is not None:
        cfg.LR = float(args.lr)
    if args.num_workers is not None:
        cfg.NUM_WORKERS = int(args.num_workers)
    if args.base_channels is not None:
        cfg.BASE_CHANNELS = int(args.base_channels)
    if args.hidden_channels is not None:
        cfg.TEMPORAL_HIDDEN_CHANNELS = int(args.hidden_channels)
    if args.max_lead_hours is not None:
        cfg.MAX_LEAD_HOURS = int(args.max_lead_hours)
        cfg.MAX_PRED_STEPS = int(cfg.MAX_LEAD_HOURS // cfg.TIME_STEP_HOURS)
        cfg.INPUT_STEPS = cfg.MAX_PRED_STEPS + 1
    if args.no_lead_reweight:
        cfg.USE_LEAD_REWEIGHT = False
    if args.smooth_loss_weight is not None:
        cfg.SMOOTH_LOSS_WEIGHT = float(args.smooth_loss_weight)
    if args.coord_loss_weight is not None:
        cfg.COORD_LOSS_WEIGHT = float(args.coord_loss_weight)
    if args.early_stop_patience is not None:
        cfg.EARLY_STOP_PATIENCE = int(args.early_stop_patience)
    if args.no_anchor_prior_decode:
        cfg.USE_ANCHOR_DISTANCE_PRIOR_DECODE = False
    if args.prior_min_sigma_deg is not None:
        cfg.PRIOR_MIN_SIGMA_DEG = float(args.prior_min_sigma_deg)
    if args.prior_sigma_per_hour is not None:
        cfg.PRIOR_SIGMA_PER_HOUR = float(args.prior_sigma_per_hour)
    if args.prior_power is not None:
        cfg.PRIOR_POWER = float(args.prior_power)
    if args.prior_hard_radius_deg is not None:
        cfg.PRIOR_HARD_RADIUS_DEG = float(args.prior_hard_radius_deg)
    if args.local_radius_deg is not None:
        cfg.LOCAL_RADIUS_DEG = float(args.local_radius_deg)
        cfg._update_local_grid_sizes()
    if args.local_radius_0_24_deg is not None:
        cfg.LOCAL_RADIUS_0_24_DEG = float(args.local_radius_0_24_deg)
    if args.local_radius_24_48_deg is not None:
        cfg.LOCAL_RADIUS_24_48_DEG = float(args.local_radius_24_48_deg)
    if args.local_radius_48_72_deg is not None:
        cfg.LOCAL_RADIUS_48_72_DEG = float(args.local_radius_48_72_deg)
    # Ensure the fixed crop radius is at least as large as the largest effective radius.
    max_eff_radius = max(float(cfg.LOCAL_RADIUS_0_24_DEG), float(cfg.LOCAL_RADIUS_24_48_DEG), float(cfg.LOCAL_RADIUS_48_72_DEG))
    if float(cfg.LOCAL_RADIUS_DEG) < max_eff_radius:
        cfg.LOCAL_RADIUS_DEG = max_eff_radius
        cfg._update_local_grid_sizes()
    if args.local_loss_weight is not None:
        cfg.LOCAL_LOSS_WEIGHT = float(args.local_loss_weight)
    if args.local_fusion_max_lead_hours is not None:
        cfg.LOCAL_FUSION_MAX_LEAD_HOURS = int(args.local_fusion_max_lead_hours)
    if args.local_min_conf is not None:
        cfg.LOCAL_MIN_CONF = float(args.local_min_conf)
        cfg.LOCAL_MIN_CONF_0_24 = float(args.local_min_conf)
        cfg.LOCAL_MIN_CONF_24_48 = float(args.local_min_conf)
        cfg.LOCAL_MIN_CONF_48_72 = float(args.local_min_conf)
    if args.local_min_conf_0_24 is not None:
        cfg.LOCAL_MIN_CONF_0_24 = float(args.local_min_conf_0_24)
    if args.local_min_conf_24_48 is not None:
        cfg.LOCAL_MIN_CONF_24_48 = float(args.local_min_conf_24_48)
    if args.local_min_conf_48_72 is not None:
        cfg.LOCAL_MIN_CONF_48_72 = float(args.local_min_conf_48_72)
    if args.no_rh700:
        cfg.USE_RH700 = False
    if args.no_ttr:
        cfg.USE_TTR = False
    if args.no_vws:
        cfg.USE_VWS = False
    if args.no_uv925:
        cfg.USE_UV925 = False
    if args.no_mslp:
        cfg.USE_MSLP = False
    return cfg


def load_or_build_index(cfg: Config, force_rebuild: bool = False) -> pd.DataFrame:
    if force_rebuild or not os.path.exists(cfg.SEQUENCE_INDEX_CSV):
        return build_sequence_index(cfg)
    print(f"Loaded sequence index: {cfg.SEQUENCE_INDEX_CSV}")
    return read_sequence_index(cfg)


def main() -> None:
    args = parse_args()
    cfg = apply_args(Config(), args)
    ensure_main_dirs(cfg)
    print(f"Output dir       : {cfg.OUT_DIR}")
    print(f"Global domain    : lat {cfg.DOMAIN_LAT_MIN}-{cfg.DOMAIN_LAT_MAX}, lon {cfg.DOMAIN_LON_MIN}-{cfg.DOMAIN_LON_MAX}")
    print(f"Input grid       : {cfg.DOMAIN_H} x {cfg.DOMAIN_W}")
    print(f"Low output grid  : {cfg.LOW_H} x {cfg.LOW_W}")
    print(f"Input steps      : {cfg.INPUT_STEPS} (t to t-{cfg.MAX_LEAD_HOURS}h)")
    print(f"Output slots     : {cfg.MAX_PRED_STEPS} (t-3h to t-{cfg.MAX_LEAD_HOURS}h)")
    print(f"Local crop grid  : {cfg.LOCAL_H} x {cfg.LOCAL_W}; local low grid {cfg.LOCAL_LOW_H} x {cfg.LOCAL_LOW_W}")
    print(f"Local valid radii: 0-24h={cfg.LOCAL_RADIUS_0_24_DEG:.1f}°, 24-48h={cfg.LOCAL_RADIUS_24_48_DEG:.1f}°, 48-72h={cfg.LOCAL_RADIUS_48_72_DEG:.1f}°")
    print(f"Train years      : {cfg.TRAIN_YEARS[0]}-{cfg.TRAIN_YEARS[1]}")
    print(f"Val years        : {cfg.VAL_YEARS[0]}-{cfg.VAL_YEARS[1]}")

    if args.mode == "build_index":
        build_sequence_index(cfg)
        return

    index_df = load_or_build_index(cfg, force_rebuild=args.force_rebuild_index)
    print("Split counts:", index_df["SPLIT"].value_counts(dropna=False).to_dict() if not index_df.empty else {})

    if args.mode == "train":
        train_model(cfg, index_df)
    elif args.mode == "validate":
        validate_model(cfg, index_df, checkpoint=args.checkpoint or cfg.BEST_CKPT)
    elif args.mode == "all":
        train_model(cfg, index_df)
        validate_model(cfg, index_df, checkpoint=cfg.BEST_CKPT)


if __name__ == "__main__":
    main()

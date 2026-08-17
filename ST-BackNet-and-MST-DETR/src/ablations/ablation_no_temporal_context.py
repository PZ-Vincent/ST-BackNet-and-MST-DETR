# -*- coding: utf-8 -*-
"""MST-DETR ablation without temporal context.

This model retains GridSat–ERA5 gated fusion and the DETR heads but removes the
3D-CNN temporal-context branch. Detection uses only the fused feature from the
current time step; cyclic time features remain available to the lead-time head.
"""

import os
import time
from pathlib import Path

# Must be set before importing torch for reproducibility.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models
from torchvision.models import swin_t, Swin_T_Weights

from scipy.optimize import linear_sum_assignment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = Path(os.environ.get("TC_DATA_ROOT", REPOSITORY_ROOT / "data" / "raw"))
OUTPUT_ROOT = Path(os.environ.get("TC_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))
DETR_HUB_REPOSITORY = "facebookresearch/detr:29901c51d7fe8712168b8d0d64351170bc0f83e0"


# ================= 1. Configuration =================
class Config:
    # ---------------- Output ----------------
    OUTPUT_DIR = str(OUTPUT_ROOT / "training" / "ablation_no_temporal_context")

    # ---------------- Input data ----------------
    DIR_GRIDSAT = str(RAW_DATA_ROOT / "gridsat" / "western_north_pacific")

    # ERA5 WP KEEP28 normalized directories:
    #   pressure_normalized raw: 20 channels
    #       u[200,500,850,925], v[200,500,850,925],
    #       t[200,500,850,925], w[200,500,850,925], q[200,500,850,925]
    #   this script keeps only t500 and t850 from the t block,
    #   so pressure channels become 18 instead of 20.
    #   single_normalized: 4 channels
    #   pressure_others_normalized: 4 channels
    #   rh700_vws200_850_normalized: 2 channels [RH700, VWS200_850]
    DIR_ERA5_PRESSURE = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "pressure_normalized")
    DIR_ERA5_SINGLE = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "single_normalized")
    DIR_ERA5_PRESSURE_OTHERS = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "pressure_others_normalized")
    DIR_ERA5_RH700_VWS200_850 = str(RAW_DATA_ROOT / "era5" / "western_north_pacific" / "rh700_vws200_850_normalized")

    # pressure_keep = u(4) + v(4) + t500/t850(2) + w(4) + q(4) = 18
    # total ERA5 = pressure_keep18 + single4 + pressure_others4 + rh/vws2 = 28
    ERA5_IN_CHANNELS = 28

    # pressure_normalized raw channel order:
    #   u[200,500,850,925] -> 0,1,2,3
    #   v[200,500,850,925] -> 4,5,6,7
    #   t[200,500,850,925] -> 8,9,10,11
    #   w[200,500,850,925] -> 12,13,14,15
    #   q[200,500,850,925] -> 16,17,18,19
    # Keep all u/v/w/q and only t500/t850.
    PRESSURE_KEEP_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
    PRESSURE_KEEP_NAMES = [
        "u200", "u500", "u850", "u925",
        "v200", "v500", "v850", "v925",
        "t500", "t850",
        "w200", "w500", "w850", "w925",
        "q200", "q500", "q850", "q925",
    ]

    POS_CSV_PATH = str(
        REPOSITORY_ROOT / "data" / "training_labels" / "western_north_pacific" / "DB_Pre_Genesis_Train.csv"
    )
    # ---------------- CSV columns ----------------
    COL_DATA_SOURCE = "DATA_SOURCE"
    COL_TIME = "ISO_TIME"
    COL_LAT = "USA_LAT"
    COL_LON = "USA_LON"
    COL_HOURS_TO_TD = "HOURS_TO_GENESIS"
    COL_SID = "SID"

    # ---------------- Training split ----------------
    # Strict current-only ablation: no historical frame is loaded.
    SEQ_LEN = 1
    TIME_INTERVAL_HOURS = 3
    MAX_HOURS_TO_TD = 72.0

    Train_years = list(range(2000, 2018))
    Val_years = list(range(2018, 2020))
    Test_years = list(range(2020, 2024))

    # ---------------- WP detection domain ----------------
    LON_MIN, LON_MAX = 100.0, 180.0
    LAT_MIN, LAT_MAX = 0.0, 40.0
    BOX_SIZE_DEG = 8.0

    # ---------------- Optimization ----------------
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    SAT_BACKBONE_LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    EVAL_BATCH_SIZE = 16
    TOP_K_SAVE = 10

    SEED_LIST = [42]

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Training is the safe default; test-only evaluation requires a selected checkpoint.
    Mode = "train_and_test"

    # Used only when Mode="test". Replace with a real checkpoint path.
    TEST_CHECKPOINT_PATH = ""
    csv_save_name = "eval_model"

    # ---------------- Unified comparison CSV output ----------------
    MODEL_NAME = "MST_DETR_NoTemporalContext"
    COMPARISON_CSV_NAME = "model_comparison_metrics_storm.csv"
    COMPARISON_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    COMPARISON_IOU_THRESHOLDS = [0.1, 0.3, 0.5]
    MAIN_CONF_THRESHOLD = 0.5
    MAIN_IOU_THRESHOLD = 0.3

    CURRENT_SEED = None
    CURRENT_OUTPUT_DIR = None
    MODEL_DIR = None


# ================= 2. Utilities =================
def configure_output_dirs_for_seed(seed: int) -> None:
    Config.CURRENT_SEED = seed
    Config.CURRENT_OUTPUT_DIR = os.path.join(Config.OUTPUT_DIR, "seed_{}".format(seed))
    Config.MODEL_DIR = os.path.join(Config.CURRENT_OUTPUT_DIR, "models")


def resolve_test_checkpoint_path(seed: int) -> str:
    """Return an explicit checkpoint or the validation-selected alias for this seed."""
    if Config.TEST_CHECKPOINT_PATH:
        return Config.TEST_CHECKPOINT_PATH
    return os.path.join(Config.MODEL_DIR, "best_model_seed{}.pth".format(seed))


def create_output_dirs() -> None:
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CURRENT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)


def check_required_paths() -> None:
    path_items = {
        "DIR_GRIDSAT": Config.DIR_GRIDSAT,
        "DIR_ERA5_PRESSURE": Config.DIR_ERA5_PRESSURE,
        "DIR_ERA5_SINGLE": Config.DIR_ERA5_SINGLE,
        "DIR_ERA5_PRESSURE_OTHERS": Config.DIR_ERA5_PRESSURE_OTHERS,
        "DIR_ERA5_RH700_VWS200_850": Config.DIR_ERA5_RH700_VWS200_850,
        "POS_CSV_PATH": Config.POS_CSV_PATH,
    }
    print("\n[WP path check]")
    for name, path in path_items.items():
        print("  {:<30} {} | {}".format(name, "OK" if os.path.exists(path) else "MISSING", path))


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def pad_to_multiple(x: torch.Tensor, k: int = 32) -> torch.Tensor:
    h, w = x.shape[-2:]
    new_h = (h // k + 1) * k if h % k != 0 else h
    new_w = (w // k + 1) * k if w % k != 0 else w
    return F.pad(x, (0, new_w - w, 0, new_h - h), value=0)


def pixel_boxes_to_lonlat_centers(boxes_xyxy: torch.Tensor, img_width: int, img_height: int) -> List[Tuple[float, float]]:
    centers = []
    if len(boxes_xyxy) == 0:
        return centers
    for box in boxes_xyxy:
        x0, y0, x1, y1 = [float(v) for v in box]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        lon = Config.LON_MIN + cx / float(img_width) * (Config.LON_MAX - Config.LON_MIN)
        lat = Config.LAT_MAX - cy / float(img_height) * (Config.LAT_MAX - Config.LAT_MIN)
        centers.append((lat, lon))
    return centers


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(np.isfinite(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan")
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return r * c


def get_lead_bin(hours_to_td: float) -> str:
    if hours_to_td < 0:
        return "unknown"
    if hours_to_td <= 24:
        return "0-24h"
    if hours_to_td <= 48:
        return "24-48h"
    if hours_to_td <= 72:
        return "48-72h"
    return ">72h"


def make_metric_bucket() -> Dict[str, Any]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "time_errors": [],
        "raw_diffs": [],
        "center_errors_km": [],
        "matched_ious": [],
        "event_total": set(),
        "event_hit_leads": {},
    }


# ================= 4. Fusion modules =================
class GatedAFM(nn.Module):
    def __init__(self, sat_dim: int, era_dim: int, out_dim: int):
        super().__init__()

        if sat_dim == out_dim:
            self.sat_proj = nn.Identity()
        else:
            self.sat_proj = nn.Sequential(
                nn.Conv2d(sat_dim, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_dim),
                nn.ReLU(inplace=True),
            )

        self.era_proj = nn.Sequential(
            nn.Conv2d(era_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True),
        )

        self.sat_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, sat_feat: torch.Tensor, era_feat: torch.Tensor) -> torch.Tensor:
        sat_feat = self.sat_proj(sat_feat)
        era_feat = self.era_proj(era_feat)
        era_feat_up = F.interpolate(era_feat, size=sat_feat.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([sat_feat, era_feat_up], dim=1)
        gate_sat = self.sat_gate(combined)
        fused = era_feat_up + gate_sat * sat_feat
        return self.out_conv(fused)


class SatPreProcessor(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


# ================= 5. Model architecture =================
class MultiModalBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.sat_pre = SatPreProcessor(1, 3)

        res18 = models.resnet18(weights=None)
        self.era_backbone = nn.Sequential(
            nn.Conv2d(Config.ERA5_IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False),
            res18.bn1,
            res18.relu,
            res18.maxpool,
            res18.layer1,
            res18.layer2,
        )

        full_swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.sat_backbone = nn.Sequential(*list(full_swin.features.children())[:6])
        self.sat_norm = nn.LayerNorm(384)
        self.sat_adapter = nn.Conv2d(384, hidden_dim, kernel_size=1)
        self.fusion = GatedAFM(sat_dim=hidden_dim, era_dim=128, out_dim=hidden_dim)

    def forward(self, sat_batch: torch.Tensor, era_batch: torch.Tensor) -> torch.Tensor:
        sat_small = self.sat_pre(sat_batch)
        x_sat = self.sat_backbone(sat_small)
        x_sat = self.sat_norm(x_sat).permute(0, 3, 1, 2)
        sat_feat = self.sat_adapter(x_sat)
        era_feat = self.era_backbone(era_batch)
        return self.fusion(sat_feat, era_feat)


class SpatioTemporalDETR(nn.Module):
    def __init__(self, num_classes: int = 1, num_frames: int = 4):
        super().__init__()
        self.hidden_dim = 256
        self.num_frames = int(num_frames)
        self.backbone = MultiModalBackbone(self.hidden_dim)

        # Remove the 3D-CNN spatiotemporal-context branch entirely.

        detr_base = torch.hub.load(DETR_HUB_REPOSITORY, "detr_resnet50", pretrained=True)
        self.transformer = detr_base.transformer
        self.query_embed = detr_base.query_embed
        self.bbox_embed = detr_base.bbox_embed
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)

        self.time_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32))
        self.time_head = nn.Sequential(nn.Linear(self.hidden_dim + 32, 128), nn.ReLU(), nn.Linear(128, 1))

    def get_spatial_pos_embed(self, bs: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        y_embed = torch.arange(h, device=device).unsqueeze(1).repeat(1, w).float() / max(h - 1, 1) * 2 * math.pi
        x_embed = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1).float() / max(w - 1, 1) * 2 * math.pi
        dim_t = torch.arange(half_dim, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_dim)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)
        return pos

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sat = inputs["sat"]
        era_keep = inputs["era_keep"]
        B, T, _, H, W = sat.shape

        sat_flat = sat.flatten(0, 1)
        era_stack = era_keep.flatten(0, 1)
        feat = self.backbone(sat_flat, era_stack)

        _, C, h_f, w_f = feat.shape
        feat = feat.view(B, T, C, h_f, w_f)

        if T != self.num_frames:
            raise RuntimeError("Input T={} does not match model num_frames={}.".format(T, self.num_frames))

        # Use only the current-time fused multimodal feature.
        # Dataset order is [current, t-3 h, ...]; index 0 is the current frame.
        curr_feat = feat[:, 0]  # [B, C, Hf, Wf]
        src_fused = curr_feat

        if not torch.isfinite(src_fused).all():
            raise RuntimeError("src_fused contains NaN/Inf in current-only fusion.")

        pos_spatial = self.get_spatial_pos_embed(B, h_f, w_f, src_fused.device)
        mask = torch.zeros((B, h_f, w_f), dtype=torch.bool, device=src_fused.device)

        hs = self.transformer(src_fused, mask, self.query_embed.weight, pos_spatial)[0]
        hs_last = hs[-1]

        outputs_class = self.class_embed(hs_last)
        outputs_coord = self.bbox_embed(hs_last).sigmoid()

        time_current = inputs["time_cyclic"][:, 0, :]
        time_emb = self.time_mlp(time_current).unsqueeze(1).expand(-1, hs_last.shape[1], -1)
        combined_feat = torch.cat([hs_last, time_emb], dim=-1)
        raw_time = self.time_head(combined_feat).squeeze(-1)
        outputs_time = torch.sigmoid(raw_time) * Config.MAX_HOURS_TO_TD

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord, "pred_time": outputs_time}


# ================= 6. Dataset =================
def get_cyclic_time(dt: datetime) -> np.ndarray:
    m = dt.month - 1
    m_sin = math.sin(2 * math.pi * m / 12)
    m_cos = math.cos(2 * math.pi * m / 12)
    h = dt.hour
    h_sin = math.sin(2 * math.pi * h / 24)
    h_cos = math.cos(2 * math.pi * h / 24)
    return np.array([m_sin, m_cos, h_sin, h_cos], dtype=np.float32)


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, pos_csv_path: str, year_range: List[int], mode: str = "train", only_official: bool = True):
        self.seq_len = Config.SEQ_LEN
        self.time_interval = Config.TIME_INTERVAL_HOURS
        self.only_official = only_official
        self.year_set = set(year_range)

        self.dir_sat = Config.DIR_GRIDSAT
        self.dir_era_p = Config.DIR_ERA5_PRESSURE
        self.dir_era_s = Config.DIR_ERA5_SINGLE
        self.dir_era_po = Config.DIR_ERA5_PRESSURE_OTHERS
        self.dir_era_rh_vws = Config.DIR_ERA5_RH700_VWS200_850
        self.time_records: Dict[str, Dict[str, Any]] = {}

        print("\n[{}] Init dataset | years={} | only_official={}".format(mode.upper(), year_range, only_official))

        if not os.path.exists(pos_csv_path):
            print("  -> [labels] ERROR: file not found {}".format(pos_csv_path))
            self.sample_keys = []
            return

        df = pd.read_csv(pos_csv_path)
        df.columns = [c.strip() for c in df.columns]
        if Config.COL_TIME not in df.columns:
            raise ValueError("POS_CSV is missing {}".format(Config.COL_TIME))
        df[Config.COL_TIME] = pd.to_datetime(df[Config.COL_TIME], format="mixed", errors="coerce")
        df = df.dropna(subset=[Config.COL_TIME]).copy()
        df = df[df[Config.COL_TIME].dt.year.isin(year_range)].copy()

        required_cols = [Config.COL_TIME, Config.COL_LAT, Config.COL_LON]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError("POS_CSV missing required column: {}".format(col))

        df[Config.COL_LAT] = pd.to_numeric(df[Config.COL_LAT], errors="coerce")
        df[Config.COL_LON] = pd.to_numeric(df[Config.COL_LON], errors="coerce")
        if Config.COL_HOURS_TO_TD in df.columns:
            df[Config.COL_HOURS_TO_TD] = pd.to_numeric(df[Config.COL_HOURS_TO_TD], errors="coerce")
        else:
            df[Config.COL_HOURS_TO_TD] = -1.0

        # ROI and target lead filter.
        df = df[
            df[Config.COL_LAT].between(Config.LAT_MIN, Config.LAT_MAX)
            & df[Config.COL_LON].between(Config.LON_MIN, Config.LON_MAX)
            & df[Config.COL_HOURS_TO_TD].between(0.0, Config.MAX_HOURS_TO_TD)
        ].copy()

        print("  -> [CSV] after year + NA ROI + 0-72h filter: {}".format(len(df)))
        if Config.COL_DATA_SOURCE in df.columns:
            print("  -> [CSV] DATA_SOURCE before official filter:")
            print(df[Config.COL_DATA_SOURCE].value_counts(dropna=False))

        if self.only_official and Config.COL_DATA_SOURCE in df.columns:
            df = df[df[Config.COL_DATA_SOURCE].astype(str).str.startswith("Official")].copy()
            print("  -> [CSV] after only_official=True: {}".format(len(df)))
            print("  -> [CSV] DATA_SOURCE after official filter:")
            print(df[Config.COL_DATA_SOURCE].value_counts(dropna=False))

        df["file_key"] = df[Config.COL_TIME].dt.strftime("%Y%m%d%H")
        grouped = df.groupby("file_key")

        pos_count = 0
        skipped_leak_count = 0
        skipped_missing_sat_count = 0

        for key, group in grouped:
            if not self._sequence_year_valid(key):
                skipped_leak_count += 1
                continue
            if not os.path.exists(os.path.join(self.dir_sat, "{}.npy".format(key))):
                skipped_missing_sat_count += 1
                continue

            targets = []
            for _, row in group.iterrows():
                lon = float(row[Config.COL_LON])
                lat = float(row[Config.COL_LAT])
                time_to_td = float(row.get(Config.COL_HOURS_TO_TD, -1.0))
                sid = row.get(Config.COL_SID, "UNKNOWN")
                targets.append((lon, lat, time_to_td, sid))

            if len(targets) == 0:
                continue
            self.time_records[key] = {"targets": targets, "type": "pos"}
            pos_count += 1

        print("  -> [positive] loaded times: {}".format(pos_count))
        print("  -> [positive] skipped by cross-year sequence: {}".format(skipped_leak_count))
        print("  -> [positive] skipped by missing GridSat: {}".format(skipped_missing_sat_count))

        self.sample_keys = list(self.time_records.keys())
        if mode == "train":
            random.shuffle(self.sample_keys)

        print("  => done. total sample times: {}\n".format(len(self.sample_keys)))

    def _sequence_year_valid(self, end_time_str: str) -> bool:
        try:
            current_dt = datetime.strptime(end_time_str, "%Y%m%d%H")
        except Exception:
            return False
        for i in range(self.seq_len):
            dt = current_dt - timedelta(hours=self.time_interval * i)
            if dt.year not in self.year_set:
                return False
        return True

    @staticmethod
    def _load_npy(path: str, expected_channels: Optional[int] = None) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            return None
        try:
            d = np.load(path).astype(np.float32)
        except Exception:
            return None
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if d.ndim == 2:
            d = d[np.newaxis, :, :]
        elif d.ndim == 3:
            # Support [H,W,C] saved arrays.
            if expected_channels is not None and d.shape[-1] == expected_channels and d.shape[0] != expected_channels:
                d = np.transpose(d, (2, 0, 1))
        else:
            return None
        if expected_channels is not None and d.shape[0] != expected_channels:
            print("[Warning] channel mismatch: {} expected {}, got {}, shape={}".format(path, expected_channels, d.shape[0], d.shape))
            return None
        return d

    def load_files(self, t_str: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sat = self._load_npy(os.path.join(self.dir_sat, "{}.npy".format(t_str)))
        era_p = self._load_npy(os.path.join(self.dir_era_p, "{}.npy".format(t_str)), expected_channels=20)
        era_s = self._load_npy(os.path.join(self.dir_era_s, "{}.npy".format(t_str)), expected_channels=4)
        era_po = self._load_npy(os.path.join(self.dir_era_po, "{}.npy".format(t_str)), expected_channels=4)
        era_rh_vws = self._load_npy(os.path.join(self.dir_era_rh_vws, "{}.npy".format(t_str)), expected_channels=2)

        if sat is None or era_p is None or era_s is None or era_po is None or era_rh_vws is None:
            return None

        # Select pressure channels: keep u/v/w/q and only temperature levels t500/t850.
        # Raw pressure order is:
        #   u[200,500,850,925], v[200,500,850,925],
        #   t[200,500,850,925], w[200,500,850,925], q[200,500,850,925].
        # Therefore t500/t850 are indices 9 and 10; t200/t925 are dropped.
        era_p = era_p[Config.PRESSURE_KEEP_INDICES]
        if era_p.shape[0] != len(Config.PRESSURE_KEEP_INDICES):
            print("[Warning] pressure channel selection failed: expected {}, got {}, shape={}".format(
                len(Config.PRESSURE_KEEP_INDICES), era_p.shape[0], era_p.shape
            ))
            return None

        era_keep = np.concatenate([era_p, era_s, era_po, era_rh_vws], axis=0).astype(np.float32)
        if era_keep.shape[0] != Config.ERA5_IN_CHANNELS:
            print("[Warning] ERA5 channel mismatch: expected {}, got {}, shape={}".format(Config.ERA5_IN_CHANNELS, era_keep.shape[0], era_keep.shape))
            return None
        return sat, era_keep

    def get_sequence(self, end_time_str: str) -> Optional[List[np.ndarray]]:
        try:
            current_dt = datetime.strptime(end_time_str, "%Y%m%d%H")
        except Exception:
            return None
        if not self._sequence_year_valid(end_time_str):
            return None

        frames = [[], [], []]  # sat, era_keep, time_cyclic
        for i in range(self.seq_len):
            dt = current_dt - timedelta(hours=self.time_interval * i)
            t_str = dt.strftime("%Y%m%d%H")
            data = self.load_files(t_str)
            t_feat = get_cyclic_time(dt)
            if data is None:
                if len(frames[0]) > 0:
                    for j in range(2):
                        frames[j].append(frames[j][-1])
                    frames[2].append(t_feat)
                else:
                    return None
            else:
                frames[0].append(data[0])
                frames[1].append(data[1])
                frames[2].append(t_feat)
        return [np.stack(f[::-1]) for f in frames]

    def __getitem__(self, idx: int):
        if len(self.sample_keys) == 0:
            raise IndexError("Dataset is empty.")
        time_str = self.sample_keys[idx]
        record = self.time_records[time_str]
        seq_data = self.get_sequence(time_str)
        if seq_data is None:
            return None

        sat_tensor = torch.from_numpy(seq_data[0])
        padded_sat = pad_to_multiple(sat_tensor, k=32)
        inputs = {
            "sat": padded_sat,
            "era_keep": torch.from_numpy(seq_data[1]),
            "time_cyclic": torch.from_numpy(seq_data[2]),
        }
        _, _, H, W = inputs["sat"].shape
        target = self._generate_target(record["targets"], H, W)
        target["time_key"] = time_str
        return inputs, target

    def _generate_target(self, target_list: List[Tuple[Any, ...]], height: int, width: int) -> Dict[str, Any]:
        target: Dict[str, Any] = {"orig_size": torch.as_tensor([int(height), int(width)])}
        boxes, labels, time_labels, sids = [], [], [], []
        pix_w = width / (Config.LON_MAX - Config.LON_MIN)
        pix_h = height / (Config.LAT_MAX - Config.LAT_MIN)

        for item in target_list:
            if len(item) == 4:
                clon, clat, time_val, sid = item
            else:
                clon, clat, time_val = item
                sid = "UNKNOWN"
            if not (Config.LON_MIN <= clon <= Config.LON_MAX and Config.LAT_MIN <= clat <= Config.LAT_MAX):
                continue
            cx = (clon - Config.LON_MIN) / (Config.LON_MAX - Config.LON_MIN) * width
            cy = (Config.LAT_MAX - clat) / (Config.LAT_MAX - Config.LAT_MIN) * height
            w_pix = Config.BOX_SIZE_DEG * pix_w
            h_pix = Config.BOX_SIZE_DEG * pix_h
            x_min = max(0.0, min(cx - w_pix / 2.0, width - 1.0))
            y_min = max(0.0, min(cy - h_pix / 2.0, height - 1.0))
            x_max = max(0.0, min(cx + w_pix / 2.0, width))
            y_max = max(0.0, min(cy + h_pix / 2.0, height))
            if (x_max - x_min) > 1.0 and (y_max - y_min) > 1.0:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(0)
                safe_time_val = float(np.clip(time_val, 0.0, Config.MAX_HOURS_TO_TD))
                time_labels.append(safe_time_val)
                sids.append(str(sid))

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["time_to_td"] = torch.as_tensor(time_labels, dtype=torch.float32)
            target["sids"] = sids
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["time_to_td"] = torch.zeros((0,), dtype=torch.float32)
            target["sids"] = []
        return target

    def __len__(self) -> int:
        return len(self.sample_keys)


# ================= 7. Matching and losses =================
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]):
        bs = outputs["pred_logits"].shape[0]
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            return [
                (torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64))
                for _ in range(bs)
            ]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"]) > 0])
        tgt_bbox = torch.cat([v["boxes_norm"] for v in targets if len(v["boxes_norm"]) > 0])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class).view(bs, -1, len(tgt_ids)).cpu()

        indices = []
        col_offset = 0
        for i, size in enumerate(sizes):
            if size == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
            else:
                c = C[i, :, col_offset:col_offset + size]
                row_ind, col_ind = linear_sum_assignment(c.numpy())
                indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))
            col_offset += size
        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: Dict[str, torch.Tensor], eos_coef: float):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"]
        src_time = outputs["pred_time"]

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        if len(idx[0]) > 0:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices) if len(J) > 0])
            target_boxes = torch.cat([t["boxes_norm"][J] for t, (_, J) in zip(targets, indices) if len(J) > 0])
            target_times = torch.cat([t["time_to_td"][J] for t, (_, J) in zip(targets, indices) if len(J) > 0])
            src_logits_o = src_logits[idx]
            src_boxes_o = src_boxes[idx]
            src_time_o = src_time[idx]
            target_classes[idx] = target_classes_o
        else:
            target_classes_o = torch.zeros((0,), dtype=torch.int64, device=src_logits.device)
            target_boxes = torch.zeros((0, 4), dtype=torch.float32, device=src_logits.device)
            target_times = torch.zeros((0,), dtype=torch.float32, device=src_logits.device)
            src_logits_o = torch.zeros((0, src_logits.shape[-1]), dtype=src_logits.dtype, device=src_logits.device)
            src_boxes_o = torch.zeros((0, 4), dtype=src_boxes.dtype, device=src_boxes.device)
            src_time_o = torch.zeros((0,), dtype=src_time.dtype, device=src_time.device)

        loss_ce = F.cross_entropy(
            src_logits.reshape(-1, src_logits.shape[-1]),
            target_classes.reshape(-1),
            weight=self.weight_dict["loss_ce"],
        )

        if len(target_classes_o) > 0:
            loss_bbox = F.l1_loss(src_boxes_o, target_boxes, reduction="none").sum() / len(target_classes_o)
            loss_time = F.l1_loss(src_time_o, target_times, reduction="none").sum() / len(target_classes_o)
            loss_time = loss_time * 0.5
        else:
            loss_bbox = torch.tensor(0.0, device=src_logits.device)
            loss_time = torch.tensor(0.0, device=src_logits.device)
        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_time": loss_time}

    @staticmethod
    def _get_src_permutation_idx(indices):
        if len(indices) == 0 or sum(len(src) for src, _ in indices) == 0:
            return torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices) if len(src) > 0])
        src_idx = torch.cat([src for (src, _) in indices if len(src) > 0])
        return batch_idx, src_idx


def prepare_targets_for_detr(targets: List[Dict[str, Any]], img_height: int, img_width: int) -> List[Dict[str, Any]]:
    new_targets = []
    for t in targets:
        new_t = {}
        new_t["labels"] = t["labels"].to(Config.DEVICE)
        new_t["time_to_td"] = torch.clamp(t["time_to_td"].to(Config.DEVICE), 0.0, Config.MAX_HOURS_TO_TD)
        boxes = t["boxes"].to(Config.DEVICE)
        if boxes.numel() > 0:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / img_width
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / img_height
            w = (boxes[:, 2] - boxes[:, 0]) / img_width
            h = (boxes[:, 3] - boxes[:, 1]) / img_height
            boxes_norm = torch.stack([cx, cy, w, h], dim=1)
        else:
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32, device=Config.DEVICE)
        new_t["boxes_norm"] = boxes_norm
        new_t["boxes"] = boxes
        new_t["sids"] = t.get("sids", [])
        new_targets.append(new_t)
    return new_targets


# ================= 8. Metrics and result export =================
def summarize_metric_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
    csi = tp / (tp + fn + fp + 1e-6)
    pod = tp / (tp + fn + 1e-6)
    far = fp / (tp + fp + 1e-6)
    precision = 1.0 - far
    f1 = 2.0 * precision * pod / (precision + pod + 1e-6)

    errors = np.asarray(bucket["time_errors"], dtype=np.float32)
    raws = np.asarray(bucket["raw_diffs"], dtype=np.float32)
    center_errors = np.asarray(bucket["center_errors_km"], dtype=np.float32)
    matched_ious = np.asarray(bucket.get("matched_ious", []), dtype=np.float32)

    mae = float(np.mean(errors)) if len(errors) > 0 else -1.0
    bias = float(np.mean(raws)) if len(raws) > 0 else -1.0
    center_mean = float(np.mean(center_errors)) if len(center_errors) > 0 else -1.0
    center_median = float(np.median(center_errors)) if len(center_errors) > 0 else -1.0
    center_p90 = float(np.percentile(center_errors, 90)) if len(center_errors) > 0 else -1.0
    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else -1.0
    median_iou = float(np.median(matched_ious)) if len(matched_ious) > 0 else -1.0

    event_total = len(bucket["event_total"])
    event_hit = len(bucket["event_hit_leads"])
    event_pod = event_hit / (event_total + 1e-6)
    first_leads = [max(v) for v in bucket["event_hit_leads"].values() if len(v) > 0]
    first_lead_mean = float(np.mean(first_leads)) if len(first_leads) > 0 else -1.0
    first_lead_median = float(np.median(first_leads)) if len(first_leads) > 0 else -1.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "CSI": csi,
        "POD": pod,
        "FAR": far,
        "Precision": precision,
        "F1": f1,
        "MAE": mae,
        "Bias": bias,
        "CenterErrMeanKm": center_mean,
        "CenterErrMedianKm": center_median,
        "CenterErrP90Km": center_p90,
        "MeanIoU": mean_iou,
        "MedianIoU": median_iou,
        "EventTotal": event_total,
        "EventHit": event_hit,
        "EventPOD": event_pod,
        "FirstLeadMeanH": first_lead_mean,
        "FirstLeadMedianH": first_lead_median,
    }


def evaluate_multithreshold(model: nn.Module, data_loader, device: torch.device, thresholds: Optional[List[float]] = None, save_csv: bool = False):
    if thresholds is None:
        thresholds = [0.5]

    model.eval()
    iou_match_threshold = 0.3
    lead_bins = ["0-24h", "24-48h", "48-72h", ">72h", "unknown"]

    stats = {}
    for th in thresholds:
        stats[th] = make_metric_bucket()
        stats[th]["lead_bins"] = {b: make_metric_bucket() for b in lead_bins}

    print("Evaluating (N_batch={}, batch_size={}), IOU={}...".format(len(data_loader), getattr(data_loader, "batch_size", "unknown"), iou_match_threshold))

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            inputs, targets = batch
            for k in inputs:
                inputs[k] = inputs[k].to(device, non_blocking=True)

            outputs = model(inputs)
            h, w = inputs["sat"].shape[-2:]
            batch_size = outputs["pred_logits"].shape[0]

            for bi in range(batch_size):
                all_probas = outputs["pred_logits"].softmax(-1)[bi, :, 0]
                all_boxes = outputs["pred_boxes"][bi]
                all_times = outputs["pred_time"][bi]

                gt_boxes = targets[bi]["boxes"].cpu()
                gt_times = targets[bi]["time_to_td"].cpu()
                gt_sids = targets[bi].get("sids", ["UNKNOWN"] * len(gt_boxes))

                for th in thresholds:
                    bucket = stats[th]
                    for sid in gt_sids:
                        bucket["event_total"].add(str(sid))

                    keep = all_probas > th
                    pred_boxes_norm = all_boxes[keep].detach().cpu()
                    pred_times = all_times[keep].detach().cpu()
                    if len(pred_boxes_norm) > 0:
                        scale_tensor = torch.tensor([w, h, w, h], dtype=pred_boxes_norm.dtype)
                        pred_boxes_pixel = box_cxcywh_to_xyxy(pred_boxes_norm) * scale_tensor
                    else:
                        pred_boxes_pixel = torch.zeros((0, 4), dtype=torch.float32)

                    num_preds = len(pred_boxes_pixel)
                    num_gts = len(gt_boxes)

                    for gi in range(num_gts):
                        lead_bin = get_lead_bin(float(gt_times[gi].item()))
                        stats[th]["lead_bins"][lead_bin]["event_total"].add(str(gt_sids[gi]))

                    if num_gts > 0 and num_preds > 0:
                        iou_matrix = torchvision.ops.box_iou(gt_boxes, pred_boxes_pixel)
                        row_ind, col_ind = linear_sum_assignment((-iou_matrix).numpy())
                        matched_pairs = []
                        used_gt, used_pred = set(), set()
                        for gi, pi in zip(row_ind, col_ind):
                            iou_val = float(iou_matrix[gi, pi].item())
                            if iou_val > iou_match_threshold:
                                matched_pairs.append((int(gi), int(pi)))
                                used_gt.add(int(gi))
                                used_pred.add(int(pi))

                        tp = len(matched_pairs)
                        fp = num_preds - len(used_pred)
                        fn = num_gts - len(used_gt)
                        bucket["tp"] += tp
                        bucket["fp"] += fp
                        bucket["fn"] += fn

                        gt_centers = pixel_boxes_to_lonlat_centers(gt_boxes, w, h)
                        pred_centers = pixel_boxes_to_lonlat_centers(pred_boxes_pixel, w, h)

                        for gi, pi in matched_pairs:
                            pred_t = float(pred_times[pi].item())
                            gt_t = float(gt_times[gi].item())
                            diff = pred_t - gt_t
                            abs_diff = abs(diff)
                            gt_lat, gt_lon = gt_centers[gi]
                            pred_lat, pred_lon = pred_centers[pi]
                            center_error = haversine_km(gt_lat, gt_lon, pred_lat, pred_lon)
                            sid = str(gt_sids[gi])

                            bucket["time_errors"].append(abs_diff)
                            bucket["raw_diffs"].append(diff)
                            bucket["center_errors_km"].append(center_error)
                            bucket["event_hit_leads"].setdefault(sid, []).append(gt_t)

                            lead_bin = get_lead_bin(gt_t)
                            lb = stats[th]["lead_bins"][lead_bin]
                            lb["tp"] += 1
                            lb["time_errors"].append(abs_diff)
                            lb["raw_diffs"].append(diff)
                            lb["center_errors_km"].append(center_error)
                            lb["event_hit_leads"].setdefault(sid, []).append(gt_t)

                        for gi in range(num_gts):
                            if gi not in used_gt:
                                lead_bin = get_lead_bin(float(gt_times[gi].item()))
                                stats[th]["lead_bins"][lead_bin]["fn"] += 1

                        if fp > 0:
                            if num_gts > 0:
                                sample_lead_bin = get_lead_bin(float(torch.median(gt_times).item()))
                            else:
                                sample_lead_bin = "unknown"
                            stats[th]["lead_bins"][sample_lead_bin]["fp"] += fp

                    elif num_gts > 0:
                        bucket["fn"] += num_gts
                        for gi in range(num_gts):
                            lead_bin = get_lead_bin(float(gt_times[gi].item()))
                            stats[th]["lead_bins"][lead_bin]["fn"] += 1

                    elif num_preds > 0:
                        bucket["fp"] += num_preds
                        stats[th]["lead_bins"]["unknown"]["fp"] += num_preds

    results, lead_results = [], []
    header = "\n{:<6} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<9} | {:<8} | {:<12}".format(
        "Th", "CSI", "POD", "FAR", "MAE(h)", "Bias(h)", "Cerr(km)", "EvtPOD", "FirstLead(h)"
    )
    print(header)

    for th in thresholds:
        row = summarize_metric_bucket(stats[th])
        row["Threshold"] = th
        results.append(row)
        print(
            "{:<6.1f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.2f} | {:<8.2f} | {:<9.1f} | {:<8.4f} | {:<12.2f}".format(
                th, row["CSI"], row["POD"], row["FAR"], row["MAE"], row["Bias"], row["CenterErrMeanKm"], row["EventPOD"], row["FirstLeadMeanH"]
            )
        )
        for lead_bin in lead_bins:
            lb_row = summarize_metric_bucket(stats[th]["lead_bins"][lead_bin])
            lb_row["Threshold"] = th
            lb_row["LeadBin"] = lead_bin
            lead_results.append(lb_row)

    if save_csv:
        pd.DataFrame(results).to_csv(os.path.join(Config.CURRENT_OUTPUT_DIR, "{}.csv".format(Config.csv_save_name)), index=False)
        pd.DataFrame(lead_results).to_csv(os.path.join(Config.CURRENT_OUTPUT_DIR, "{}_lead_bins.csv".format(Config.csv_save_name)), index=False)

    return {row["Threshold"]: row for row in results}


def _format_float_for_col(x: float) -> str:
    return ("{:.1f}".format(float(x))).replace(".", "p")


def _compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP as area under the interpolated precision-recall curve."""
    if recall.size == 0 or precision.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall.astype(np.float64), [1.0]))
    mpre = np.concatenate(([0.0], precision.astype(np.float64), [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def collect_eval_records(model: nn.Module, data_loader, device: torch.device) -> List[Dict[str, Any]]:
    """
    Run model once and keep all detections/ground truth in CPU memory.
    The returned records are reused for AP, PR curves, threshold curves,
    performance diagram data, lead-bin tables, localization error and time-error metrics.
    """
    model.eval()
    records: List[Dict[str, Any]] = []
    sample_index = 0

    print("Collecting predictions for unified comparison CSV (N_batch={}, batch_size={})...".format(
        len(data_loader), getattr(data_loader, "batch_size", "unknown")
    ))

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            inputs, targets = batch
            for k in inputs:
                inputs[k] = inputs[k].to(device, non_blocking=True)

            outputs = model(inputs)
            h, w = inputs["sat"].shape[-2:]
            batch_size = outputs["pred_logits"].shape[0]
            scale_tensor = torch.tensor([w, h, w, h], dtype=torch.float32)

            for bi in range(batch_size):
                scores = outputs["pred_logits"].softmax(-1)[bi, :, 0].detach().cpu().float()
                boxes_norm = outputs["pred_boxes"][bi].detach().cpu().float()
                pred_boxes_pixel = box_cxcywh_to_xyxy(boxes_norm) * scale_tensor
                pred_times = outputs["pred_time"][bi].detach().cpu().float()

                gt_boxes = targets[bi]["boxes"].detach().cpu().float()
                gt_times = targets[bi]["time_to_td"].detach().cpu().float()
                gt_sids = [str(s) for s in targets[bi].get("sids", ["UNKNOWN"] * len(gt_boxes))]

                records.append({
                    "sample_index": sample_index,
                    "img_h": int(h),
                    "img_w": int(w),
                    "time_key": targets[bi].get("time_key", ""),
                    "gt_boxes": gt_boxes,
                    "gt_times": gt_times,
                    "gt_sids": gt_sids,
                    "pred_boxes": pred_boxes_pixel,
                    "pred_scores": scores,
                    "pred_times": pred_times,
                })
                sample_index += 1

    print("Collected {} samples for metric export.".format(len(records)))
    return records


def _threshold_stats_from_records(
    records: List[Dict[str, Any]],
    confidence_threshold: float,
    iou_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    lead_bins = ["0-24h", "24-48h", "48-72h", ">72h", "unknown"]
    bucket = make_metric_bucket()
    lead_bucket = {b: make_metric_bucket() for b in lead_bins}

    for rec in records:
        gt_boxes = rec["gt_boxes"]
        gt_times = rec["gt_times"]
        gt_sids = rec["gt_sids"]
        pred_scores = rec["pred_scores"]
        pred_boxes_all = rec["pred_boxes"]
        pred_times_all = rec["pred_times"]
        h, w = rec["img_h"], rec["img_w"]

        for sid in gt_sids:
            bucket["event_total"].add(str(sid))

        for gi in range(len(gt_boxes)):
            lead_bin = get_lead_bin(float(gt_times[gi].item()))
            lead_bucket[lead_bin]["event_total"].add(str(gt_sids[gi]))

        keep = pred_scores > confidence_threshold
        pred_boxes = pred_boxes_all[keep]
        pred_times = pred_times_all[keep]

        num_gts = len(gt_boxes)
        num_preds = len(pred_boxes)

        if num_gts > 0 and num_preds > 0:
            iou_matrix = torchvision.ops.box_iou(gt_boxes, pred_boxes)
            row_ind, col_ind = linear_sum_assignment((-iou_matrix).numpy())
            matched_pairs = []
            used_gt, used_pred = set(), set()
            for gi, pi in zip(row_ind, col_ind):
                iou_val = float(iou_matrix[gi, pi].item())
                if iou_val >= iou_threshold:
                    matched_pairs.append((int(gi), int(pi), iou_val))
                    used_gt.add(int(gi))
                    used_pred.add(int(pi))

            tp = len(matched_pairs)
            fp = num_preds - len(used_pred)
            fn = num_gts - len(used_gt)
            bucket["tp"] += tp
            bucket["fp"] += fp
            bucket["fn"] += fn

            gt_centers = pixel_boxes_to_lonlat_centers(gt_boxes, w, h)
            pred_centers = pixel_boxes_to_lonlat_centers(pred_boxes, w, h)

            for gi, pi, iou_val in matched_pairs:
                pred_t = float(pred_times[pi].item())
                gt_t = float(gt_times[gi].item())
                diff = pred_t - gt_t
                abs_diff = abs(diff)
                gt_lat, gt_lon = gt_centers[gi]
                pred_lat, pred_lon = pred_centers[pi]
                center_error = haversine_km(gt_lat, gt_lon, pred_lat, pred_lon)
                sid = str(gt_sids[gi])

                bucket["time_errors"].append(abs_diff)
                bucket["raw_diffs"].append(diff)
                bucket["center_errors_km"].append(center_error)
                bucket["matched_ious"].append(iou_val)
                bucket["event_hit_leads"].setdefault(sid, []).append(gt_t)

                lead_bin = get_lead_bin(gt_t)
                lb = lead_bucket[lead_bin]
                lb["tp"] += 1
                lb["time_errors"].append(abs_diff)
                lb["raw_diffs"].append(diff)
                lb["center_errors_km"].append(center_error)
                lb["matched_ious"].append(iou_val)
                lb["event_hit_leads"].setdefault(sid, []).append(gt_t)

            for gi in range(num_gts):
                if gi not in used_gt:
                    lead_bin = get_lead_bin(float(gt_times[gi].item()))
                    lead_bucket[lead_bin]["fn"] += 1

            if fp > 0:
                if num_gts > 0:
                    sample_lead_bin = get_lead_bin(float(torch.median(gt_times).item()))
                else:
                    sample_lead_bin = "unknown"
                lead_bucket[sample_lead_bin]["fp"] += fp

        elif num_gts > 0:
            bucket["fn"] += num_gts
            for gi in range(num_gts):
                lead_bin = get_lead_bin(float(gt_times[gi].item()))
                lead_bucket[lead_bin]["fn"] += 1

        elif num_preds > 0:
            bucket["fp"] += num_preds
            lead_bucket["unknown"]["fp"] += num_preds

    return bucket, lead_bucket


def _ap_pr_from_records(records: List[Dict[str, Any]], iou_threshold: float) -> Tuple[float, List[Dict[str, Any]]]:
    total_gt = sum(len(r["gt_boxes"]) for r in records)
    detections: List[Tuple[float, int, int]] = []
    for ridx, rec in enumerate(records):
        scores = rec["pred_scores"]
        for pidx in range(len(scores)):
            detections.append((float(scores[pidx].item()), ridx, pidx))
    detections.sort(key=lambda x: x[0], reverse=True)

    if total_gt == 0 or len(detections) == 0:
        return 0.0, []

    used_gt = {ridx: set() for ridx in range(len(records))}
    tp_flags, fp_flags = [], []
    pr_rows: List[Dict[str, Any]] = []
    tp_cum = 0
    fp_cum = 0

    for rank, (score, ridx, pidx) in enumerate(detections, start=1):
        rec = records[ridx]
        gt_boxes = rec["gt_boxes"]
        pred_box = rec["pred_boxes"][pidx:pidx + 1]

        is_tp = 0
        matched_iou = -1.0
        if len(gt_boxes) > 0:
            ious = torchvision.ops.box_iou(gt_boxes, pred_box).squeeze(1)
            best_iou, best_gt = torch.max(ious, dim=0)
            best_iou_value = float(best_iou.item())
            best_gt_index = int(best_gt.item())
            if best_iou_value >= iou_threshold and best_gt_index not in used_gt[ridx]:
                is_tp = 1
                matched_iou = best_iou_value
                used_gt[ridx].add(best_gt_index)

        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precision = tp_cum / (tp_cum + fp_cum + 1e-12)
        recall = tp_cum / (total_gt + 1e-12)
        tp_flags.append(is_tp)
        fp_flags.append(1 - is_tp)
        pr_rows.append({
            "Rank": rank,
            "ScoreThreshold": score,
            "TP": tp_cum,
            "FP": fp_cum,
            "FN": total_gt - tp_cum,
            "Precision": precision,
            "Recall": recall,
            "POD": recall,
            "MatchedIoUAtRank": matched_iou,
        })

    precision_arr = np.asarray([r["Precision"] for r in pr_rows], dtype=np.float64)
    recall_arr = np.asarray([r["Recall"] for r in pr_rows], dtype=np.float64)
    ap = _compute_ap_from_pr(recall_arr, precision_arr)
    return ap, pr_rows


def _base_export_row(row_type: str, split: str, checkpoint_path: str, iou_threshold: float = -1.0,
                     confidence_threshold: float = -1.0, lead_bin: str = "all") -> Dict[str, Any]:
    return {
        "RowType": row_type,
        "ModelName": Config.MODEL_NAME,
        "Seed": Config.CURRENT_SEED,
        "Split": split,
        "CheckpointFile": os.path.basename(checkpoint_path) if checkpoint_path else "",
        "IoUThreshold": iou_threshold,
        "ConfidenceThreshold": confidence_threshold,
        "LeadBin": lead_bin,
    }


def _metric_row_from_summary(row_type: str, split: str, checkpoint_path: str, iou_threshold: float,
                             confidence_threshold: float, lead_bin: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    row = _base_export_row(row_type, split, checkpoint_path, iou_threshold, confidence_threshold, lead_bin)
    row.update(summary)
    row["Recall"] = summary.get("POD", -1.0)
    row["SuccessRatio"] = summary.get("Precision", -1.0)
    return row



def export_model_comparison_csv(
    model: nn.Module,
    data_loader,
    device: torch.device,
    split: str,
    checkpoint_path: str,
    save_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export one unified CSV for cross-model comparison.

    CSV row types:
      - summary_main: one row for the main table, using MAIN_IOU_THRESHOLD and MAIN_CONF_THRESHOLD.
      - ap_summary: AP for each IoU threshold.
      - pr_curve: ranked precision-recall records.
      - threshold_overall: CSI/POD/FAR/F1/localization/time metrics at confidence thresholds 0.1-0.9.
      - threshold_lead_bin: the same metrics stratified by lead-time bins.
    """
    if save_name is None:
        save_name = Config.COMPARISON_CSV_NAME

    records = collect_eval_records(model, data_loader, device)
    all_rows: List[Dict[str, Any]] = []
    ap_by_iou: Dict[float, float] = {}
    threshold_summary_by_key: Dict[Tuple[float, float], Dict[str, Any]] = {}

    for iou_thr in Config.COMPARISON_IOU_THRESHOLDS:
        print("Computing AP/PR data | split={} | IoU={:.2f}".format(split, iou_thr))
        ap, pr_rows = _ap_pr_from_records(records, iou_thr)
        ap_by_iou[float(iou_thr)] = ap

        ap_row = _base_export_row("ap_summary", split, checkpoint_path, iou_thr, -1.0, "all")
        ap_row.update({"AP": ap, "NumPRPoints": len(pr_rows)})
        all_rows.append(ap_row)

        for r in pr_rows:
            row = _base_export_row("pr_curve", split, checkpoint_path, iou_thr, -1.0, "all")
            row.update(r)
            row["AP"] = ap
            all_rows.append(row)

    for iou_thr in Config.COMPARISON_IOU_THRESHOLDS:
        for conf_thr in Config.COMPARISON_THRESHOLDS:
            print("Computing threshold metrics | split={} | IoU={:.2f} | conf={:.1f}".format(split, iou_thr, conf_thr))
            bucket, lead_bucket = _threshold_stats_from_records(records, conf_thr, iou_thr)
            summary = summarize_metric_bucket(bucket)
            threshold_summary_by_key[(float(iou_thr), float(conf_thr))] = summary
            all_rows.append(_metric_row_from_summary(
                "threshold_overall", split, checkpoint_path, iou_thr, conf_thr, "all", summary
            ))
            for lead_bin, lb in lead_bucket.items():
                lb_summary = summarize_metric_bucket(lb)
                all_rows.append(_metric_row_from_summary(
                    "threshold_lead_bin", split, checkpoint_path, iou_thr, conf_thr, lead_bin, lb_summary
                ))

    main_key = (float(Config.MAIN_IOU_THRESHOLD), float(Config.MAIN_CONF_THRESHOLD))
    main_summary = dict(threshold_summary_by_key.get(main_key, {}))
    main_row = _base_export_row(
        "summary_main", split, checkpoint_path, Config.MAIN_IOU_THRESHOLD, Config.MAIN_CONF_THRESHOLD, "all"
    )
    main_row.update(main_summary)
    main_row["Recall"] = main_summary.get("POD", -1.0)
    main_row["SuccessRatio"] = main_summary.get("Precision", -1.0)
    for iou_thr, ap_val in ap_by_iou.items():
        main_row["AP_IoU{}".format(_format_float_for_col(iou_thr))] = ap_val
    main_row["AP"] = ap_by_iou.get(float(Config.MAIN_IOU_THRESHOLD), -1.0)
    all_rows.insert(0, main_row)

    df = pd.DataFrame(all_rows)
    preferred_cols = [
        "RowType", "ModelName", "Seed", "Split", "CheckpointFile",
        "IoUThreshold", "ConfidenceThreshold", "LeadBin", "Rank", "ScoreThreshold",
        "AP", "AP_IoU0p1", "AP_IoU0p3", "AP_IoU0p5",
        "Recall", "Precision", "SuccessRatio", "CSI", "POD", "FAR", "F1",
        "TP", "FP", "FN", "MAE", "Bias",
        "CenterErrMeanKm", "CenterErrMedianKm", "CenterErrP90Km",
        "MeanIoU", "MedianIoU", "MatchedIoUAtRank",
        "EventTotal", "EventHit", "EventPOD", "FirstLeadMeanH", "FirstLeadMedianH", "NumPRPoints",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[[c for c in preferred_cols if c in df.columns] + other_cols]

    save_path = os.path.join(Config.CURRENT_OUTPUT_DIR, save_name)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("Unified comparison CSV saved to: {}".format(save_path))
    print("Main summary row:")
    print(pd.DataFrame([main_row]))
    return main_row


# ================= 9. Reproducibility and loops =================
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return ({k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}, [b[1] for b in batch])


def run_single_seed(seed: int) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("Starting run with random seed = {}".format(seed))
    print("=" * 80)

    configure_output_dirs_for_seed(seed)
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    create_output_dirs()
    check_required_paths()
    print("--- MST_DETR_NoTemporalContext: DETR uses the current-time fused feature without the 3D-CNN context path. | seed={} ---".format(seed))
    print("ERA5 pressure keep indices:", Config.PRESSURE_KEEP_INDICES)
    print("ERA5 pressure keep names  :", Config.PRESSURE_KEEP_NAMES)
    print("ERA5 total channels       :", Config.ERA5_IN_CHANNELS)
    print("Ablation setting          : Remove the 3D-CNN spatiotemporal context path; DETR uses current-time fused feature only.")

    summary_row = {
        "seed": seed,
        "best_model_file": "",
        "best_val_csi": -1,
        "test_csi_0.5": -1,
        "test_pod_0.5": -1,
        "test_far_0.5": -1,
        "test_mae_0.5": -1,
        "test_bias_0.5": -1,
        "test_center_err_mean_km_0.5": -1,
        "test_center_err_median_km_0.5": -1,
        "test_event_pod_0.5": -1,
        "test_first_lead_mean_h_0.5": -1,
        "test_ap_iou0.3": -1,
        "comparison_csv": "",
    }

    if Config.Mode == "train_and_test":
        dataset_train = MultiModalDataset(Config.POS_CSV_PATH, Config.Train_years, mode="train", only_official=False)
        if len(dataset_train) == 0:
            print("[Error] training set is empty.")
            return summary_row

        g = torch.Generator()
        g.manual_seed(seed)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
        dataset_val = MultiModalDataset(Config.POS_CSV_PATH, Config.Val_years, mode="val", only_official=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=Config.EVAL_BATCH_SIZE, collate_fn=collate_fn)

        if len(dataset_val) == 0:
            raise RuntimeError("Validation set is empty; checkpoint selection cannot proceed.")

        model = SpatioTemporalDETR(num_classes=1, num_frames=Config.SEQ_LEN).to(Config.DEVICE)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "sat_backbone" in n], "lr": Config.SAT_BACKBONE_LEARNING_RATE},
            {"params": [p for n, p in model.named_parameters() if "sat_backbone" not in n], "lr": Config.LEARNING_RATE},
        ]
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        class_weights = torch.ones(2).to(Config.DEVICE)
        class_weights[1] = 0.1
        weight_dict_for_sum = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_time": 0.1}
        criterion = SetCriterion(1, HungarianMatcher(), {"loss_ce": class_weights}, eos_coef=0.2).to(Config.DEVICE)

        print("\nStart Training...")
        top_k_models: List[Tuple[float, int, str]] = []

        for epoch in range(Config.NUM_EPOCHS):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            step_count = 0

            for i, batch in enumerate(data_loader_train):
                if batch is None:
                    continue
                inputs, targets = batch
                for k in inputs:
                    inputs[k] = inputs[k].to(Config.DEVICE)
                h, w = inputs["sat"].shape[-2:]
                targets = prepare_targets_for_detr(targets, h, w)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * weight_dict_for_sum[k] for k in loss_dict if k in weight_dict_for_sum)
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                total_loss += float(losses.item())
                step_count += 1
                if i % 20 == 0:
                    print("  Seed {} | Ep {} Step {}, Loss: {:.4f}".format(seed, epoch + 1, i, losses.item()))

            scheduler.step()
            avg_loss = total_loss / max(step_count, 1)
            epoch_train_time = time.time() - epoch_start_time
            print("Seed {} | Epoch {} Done. Avg Loss: {:.4f} | Train Time: {:.2f}s ({:.2f}min)".format(seed, epoch + 1, avg_loss, epoch_train_time, epoch_train_time / 60.0))

            print("\n[Validation evaluation | 2018-2019 | seed={}]".format(seed))
            res_val = evaluate_multithreshold(model, data_loader_val, Config.DEVICE, thresholds=[0.5]) if len(dataset_val) > 0 else {0.5: {"CSI": -1, "POD": -1}}
            val_csi = float(res_val[0.5]["CSI"])
            val_pod = float(res_val[0.5]["POD"])

            if len(top_k_models) < Config.TOP_K_SAVE or val_csi > top_k_models[-1][0]:
                ckpt_name = "checkpoint_seed{}_epoch{:03d}.pth".format(seed, epoch + 1)
                save_path = os.path.join(Config.MODEL_DIR, ckpt_name)
                torch.save(model.state_dict(), save_path)
                print("  ==> saved: {}".format(ckpt_name))


                top_k_models.append((val_csi, epoch + 1, save_path))
                top_k_models.sort(key=lambda x: x[0], reverse=True)
                if len(top_k_models) > Config.TOP_K_SAVE:
                    removed = top_k_models.pop()
                    if os.path.exists(removed[2]):
                        os.remove(removed[2])
                if top_k_models and top_k_models[0][2] == save_path:
                    torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, "best_model_seed{}.pth".format(seed)))

        if len(top_k_models) > 0:
            # Construct the test split only after validation-based checkpoint selection.
            dataset_test = MultiModalDataset(
                Config.POS_CSV_PATH, Config.Test_years, mode="test", only_official=True
            )
            if len(dataset_test) == 0:
                print("[Warning] test set is empty. Check years and DATA_SOURCE.")
                return summary_row
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
            )

            best_model_path = top_k_models[0][2]
            best_val_csi = top_k_models[0][0]
            summary_row["best_model_file"] = os.path.basename(best_model_path)
            summary_row["best_val_csi"] = best_val_csi

            print("\nEvaluating Best Model for seed={}: {}".format(seed, best_model_path))
            model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE, weights_only=True))
            final_res = evaluate_multithreshold(
                model,
                data_loader_test,
                Config.DEVICE,
                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                save_csv=True,
            )
            summary_row["test_csi_0.5"] = final_res[0.5]["CSI"]
            summary_row["test_pod_0.5"] = final_res[0.5]["POD"]
            summary_row["test_far_0.5"] = final_res[0.5]["FAR"]
            summary_row["test_mae_0.5"] = final_res[0.5]["MAE"]
            summary_row["test_bias_0.5"] = final_res[0.5]["Bias"]
            summary_row["test_center_err_mean_km_0.5"] = final_res[0.5]["CenterErrMeanKm"]
            summary_row["test_center_err_median_km_0.5"] = final_res[0.5]["CenterErrMedianKm"]
            summary_row["test_event_pod_0.5"] = final_res[0.5]["EventPOD"]
            summary_row["test_first_lead_mean_h_0.5"] = final_res[0.5]["FirstLeadMeanH"]

            comparison_metrics = export_model_comparison_csv(
                model,
                data_loader_test,
                Config.DEVICE,
                split="test",
                checkpoint_path=best_model_path,
            )
            summary_row["comparison_csv"] = os.path.join(Config.CURRENT_OUTPUT_DIR, Config.COMPARISON_CSV_NAME)
            if comparison_metrics:
                summary_row["test_ap_iou0.3"] = comparison_metrics.get("AP", -1)


    elif Config.Mode == "test":
        test_checkpoint_path = resolve_test_checkpoint_path(seed)
        if not os.path.exists(test_checkpoint_path):
            print("[Error] checkpoint not found: {}".format(test_checkpoint_path))
            return summary_row
        dataset_test = MultiModalDataset(Config.POS_CSV_PATH, Config.Test_years, mode="test", only_official=True)
        if len(dataset_test) == 0:
            print("[Error] test set is empty.")
            return summary_row
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
        model = SpatioTemporalDETR(num_classes=1, num_frames=Config.SEQ_LEN).to(Config.DEVICE)
        print("Loading weights: {}".format(test_checkpoint_path))
        model.load_state_dict(torch.load(test_checkpoint_path, map_location=Config.DEVICE, weights_only=True))
        print("\n=== Test evaluation (2020-2023) | seed={} ===".format(seed))
        final_res = evaluate_multithreshold(
            model,
            data_loader_test,
            Config.DEVICE,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            save_csv=True,
        )
        summary_row["best_model_file"] = os.path.basename(test_checkpoint_path)
        summary_row["test_csi_0.5"] = final_res[0.5]["CSI"]
        summary_row["test_pod_0.5"] = final_res[0.5]["POD"]
        summary_row["test_far_0.5"] = final_res[0.5]["FAR"]
        summary_row["test_mae_0.5"] = final_res[0.5]["MAE"]
        summary_row["test_bias_0.5"] = final_res[0.5]["Bias"]
        summary_row["test_center_err_mean_km_0.5"] = final_res[0.5]["CenterErrMeanKm"]
        summary_row["test_center_err_median_km_0.5"] = final_res[0.5]["CenterErrMedianKm"]
        summary_row["test_event_pod_0.5"] = final_res[0.5]["EventPOD"]
        summary_row["test_first_lead_mean_h_0.5"] = final_res[0.5]["FirstLeadMeanH"]

        comparison_metrics = export_model_comparison_csv(
            model,
            data_loader_test,
            Config.DEVICE,
            split="test",
            checkpoint_path=test_checkpoint_path,
        )
        summary_row["comparison_csv"] = os.path.join(Config.CURRENT_OUTPUT_DIR, Config.COMPARISON_CSV_NAME)
        if comparison_metrics:
            summary_row["test_ap_iou0.3"] = comparison_metrics.get("AP", -1)

    return summary_row


def main() -> None:
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    summary_rows = []
    for seed in Config.SEED_LIST:
        row = run_single_seed(seed)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(Config.OUTPUT_DIR, "multi_seed_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print("\n" + "=" * 80)
    print("multi-seed experiment finished")
    print("Summary saved to: {}".format(summary_csv_path))
    print(summary_df)
    print("=" * 80)


if __name__ == "__main__":
    main()

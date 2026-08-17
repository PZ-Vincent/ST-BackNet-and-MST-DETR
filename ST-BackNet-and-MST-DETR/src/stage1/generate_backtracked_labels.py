# -*- coding: utf-8 -*-
r"""
Generate Direct-ST lead-selected local-canvas backtracked DB_ADD dataset with P98 displacement and VO850/VWS200_850 dynamic gates
=======================================================================================

This script generates 0-72 h pre-genesis DB samples using a trained Direct ST-ResUNet-FPN
Encoder-Decoder model with lead-selected local-canvas fusion. It is designed for the
one-shot model that reads the full ERA5 sequence from TD formation time t to t-72 h and
predicts all 24 historical center slots at once.

Main logic
----------
1. Use the trained Direct-ST model to predict t-3h ... t-72h centers in one forward pass.
2. Preserve official IBTrACS DB records within 0-72 h and evaluate, but do not remove,
   official samples by environmental gate.
3. Fill missing pre-genesis DB slots with model-predicted centers.
4. Apply multiple gates to inferred samples:
   - ROI gate: 100-180E, 0-40N
   - model confidence gate
   - 3 h displacement gate relative to the previous accepted/official point
   - optional acceleration/smoothness gate
   - optional TD-anchor distance sanity gate
   - optional global-local consistency gate
   - mature TC exclusion gate
   - ERA5 dynamic/environment gates: inferred samples must satisfy vo850_core_max > official VO850 percentile and vws200_850_env_mean < official VWS percentile by default
5. Save one full backtracked CSV, one training CSV, stop-reason summary, generation summary JSON,
   and gate-threshold JSON.

Recommended usage
-----------------
python generate_backtracked_labels.py --checkpoint <path-to-st-backnet-checkpoint>

Notes
-----
- This generator is for the Direct-ST one-shot path model. It does not autoregressively
  crop around the previous prediction.
- The displacement and path gates are still evaluated sequentially from t to t-72h to
  keep the saved inferred path physically continuous.
"""

import os
import sys
import json
import math
import argparse
import importlib.util
from pathlib import Path
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = Path(os.environ.get("TC_DATA_ROOT", REPOSITORY_ROOT / "data" / "raw"))
PUBLIC_OUTPUT_ROOT = Path(os.environ.get("TC_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs"))


# =========================================================
# 1. Defaults
# =========================================================

DEFAULT_MODEL_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "train_st_backnet.py",
)

DEFAULT_CHECKPOINT = str(
    PUBLIC_OUTPUT_ROOT / "training" / "st_backnet" / "st_backnet_lead_selected_local_canvas"
    / "checkpoints" / "best_direct_st_resunet_fpn_encoder_decoder_9var.pth"
)

DEFAULT_OUT_DIR = str(PUBLIC_OUTPUT_ROOT / "generated_labels" / "western_north_pacific")

DEFAULT_IBTRACS_PATH = str(RAW_DATA_ROOT / "ibtracs" / "western_north_pacific.csv")

BACKTRACK_OUTPUT_NAME = "backtracked_labels.csv"
TRAIN_OUTPUT_NAME = "DB_Pre_Genesis_Train.csv"
STOP_SUMMARY_NAME = "generation_stop_summary.csv"
SUMMARY_JSON_NAME = "generation_summary.json"
GATE_THRESHOLD_JSON_NAME = "gate_settings.json"

TARGET_HISTORY_HOURS = 72
TIME_STEP_HOURS = 3

ROI_LAT_MIN, ROI_LAT_MAX = 0.0, 40
ROI_LON_MIN, ROI_LON_MAX = 100.0, 180.0

# Model and path gates.
DEFAULT_MIN_CONF = 0.10
DEFAULT_MAX_DISP_DEG = None      # default: estimate from official 3 h displacement P98
DEFAULT_MAX_ACCEL_DEG = 0.0      # 0 disables acceleration gate; set e.g. 1.5-2.5 to enable
DEFAULT_ANCHOR_DIST_PER_HOUR = 0.50
DEFAULT_ANCHOR_DIST_MIN_DEG = 3.0
DEFAULT_ANCHOR_DIST_BUFFER_DEG = 2.0
DEFAULT_USE_ANCHOR_DISTANCE_GATE = True
DEFAULT_GLOBAL_LOCAL_MAX_DIST_0_24 = 0.0   # 0 disables; optional values include 3.0 or 5.0 degrees.
DEFAULT_GLOBAL_LOCAL_MAX_DIST_24_48 = 0.0
DEFAULT_GLOBAL_LOCAL_MAX_DIST_48_72 = 0.0

SEARCH_RADIUS_MIN_DEG = 2.0
SEARCH_RADIUS_MAX_DEG = 2.0
DEFAULT_GATE_RADIUS_DEG = 2.0
ENV_RADIUS_FACTOR = 2.0

USE_MATURE_TC_EXCLUSION = True
MATURE_TC_MIN_WIND = 34.0
MATURE_TC_EXCLUSION_RADIUS_DEG = 5.0

BACKTRACK_YEARS = (2000, 2025)
CALIBRATION_YEARS = (2000, 2017)

FEATURE_CONFIG = {
    "vo850_core_max": {"direction": "high_good"},
    "rh700_env_mean": {"direction": "high_good"},
    "ttr_core_max": {"direction": "high_good"},
    "vws200_850_env_mean": {"direction": "low_good"},
}


# =========================================================
# Environment/path gate settings: edit here directly for new experiments
# =========================================================

DISPLACEMENT_GATE_PERCENTILE = 98.0

DISABLE_ERA5_ENVIRONMENT_HARD_GATE = False
ENV_GATE_VARIABLES = ["vo850_core_max", "vws200_850_env_mean"]
ENV_GATE_DIRECTIONS = {
    "vo850_core_max": "high_good",
    "vws200_850_env_mean": "low_good",
}
VO850_GATE_PERCENTILE = 0.02
VWS200_850_GATE_PERCENTILE = 0.98

KEEP_OFFICIAL_BEYOND72 = True
OFFICIAL_BEYOND72_MAX_HOURS: Optional[float] = None

DETECTION_REQUIRED_COLUMNS = [
    "SID",
    "ISO_TIME",
    "USA_LAT",
    "USA_LON",
    "HOURS_TO_GENESIS",
    "DATA_SOURCE",
]


def no_environment_features(reason: str = "environment_gate_disabled") -> Dict[str, Any]:
    """Return empty meteorological feature columns without reading ERA5 gate fields."""
    return {
        "valid": True,
        "reason": reason,
        "vo850_core_max": np.nan,
        "rh700_env_mean": np.nan,
        "ttr_core_max": np.nan,
        "vws200_850_env_mean": np.nan,
    }


def no_environment_gate(gate_level: str = "NoEnvGate") -> Dict[str, Any]:
    """Return an always-accepted gate result when the ERA5 environment gate is disabled."""
    return {
        "accepted": True,
        "gate_level": gate_level,
        "vo_pass": np.nan,
        "rh_pass": np.nan,
        "ttr_pass": np.nan,
        "vws_pass": np.nan,
        "env_pass_count": np.nan,
        "gate_flags": "environment_gate_disabled",
    }


# =========================================================
# 2. Basic utilities
# =========================================================

def load_module_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model script not found: {path}")
    spec = importlib.util.spec_from_file_location("direct_st_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def in_roi(lat: float, lon: float) -> bool:
    return ROI_LAT_MIN <= float(lat) <= ROI_LAT_MAX and ROI_LON_MIN <= float(lon) <= ROI_LON_MAX


def in_time_window(hours_to_genesis: float, target_history_hours: float) -> bool:
    if not np.isfinite(hours_to_genesis):
        return False
    return 0.0 < float(hours_to_genesis) <= float(target_history_hours)


def normalize_db_add_status(status: pd.Series) -> pd.Series:
    status_upper = status.astype(str).str.strip().str.upper()
    return status_upper.replace({"DB_ADD": "DB"})


def geo_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if not all(np.isfinite(v) for v in vals):
        return float("nan")
    return float(np.sqrt((float(lat1) - float(lat2)) ** 2 + (float(lon1) - float(lon2)) ** 2))


def add_stop(stop_reasons: Dict[str, int], reason: str) -> None:
    stop_reasons[reason] = stop_reasons.get(reason, 0) + 1


def safe_to_csv(df: pd.DataFrame, path: str, label: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_locked_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(fallback, index=False)
        print(f"[Warning] Cannot overwrite locked {label}: {path}")
        print(f"[Warning] Saved {label} to fallback path: {fallback}")
        return fallback


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanquantile(arr, q))


def percentile_label_from_fraction(q: float) -> str:
    """Convert a 0-1 quantile to a label such as P02 or P98."""
    return f"P{int(round(float(q) * 100)):02d}"


def percentile_label_from_percentile(p: float) -> str:
    """Convert a 0-100 percentile to a label such as P02 or P98."""
    return f"P{int(round(float(p))):02d}"


def nan_stat(arr: Optional[np.ndarray], mode: str) -> float:
    if arr is None or arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    if mode == "max":
        return float(np.nanmax(arr))
    if mode == "min":
        return float(np.nanmin(arr))
    if mode == "mean":
        return float(np.nanmean(arr))
    raise ValueError(mode)


def normalize_rh(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(x * 100.0) if 0 <= x <= 1.5 else float(x)


def make_static_fields(row: pd.Series) -> Dict[str, Any]:
    return {k: row[k] if k in row else None for k in ["SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME"]}


def lead_bin(hours: float) -> str:
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


def max_global_local_dist_for_lead(args: argparse.Namespace, lead_hour: float) -> float:
    if 0 < lead_hour <= 24:
        return float(args.global_local_max_dist_0_24)
    if 24 < lead_hour <= 48:
        return float(args.global_local_max_dist_24_48)
    if 48 < lead_hour <= 72:
        return float(args.global_local_max_dist_48_72)
    return 0.0


def anchor_distance_limit(args: argparse.Namespace, lead_hour: float) -> float:
    return max(float(args.anchor_dist_min_deg), float(args.anchor_dist_per_hour) * float(lead_hour)) + float(args.anchor_dist_buffer_deg)


# =========================================================
# 3. Environment-gate utilities
# =========================================================

# This fused version uses configurable VO850 lower-bound and VWS200_850 upper-bound gates by default.
# The settings are centralized above in the 'Environment/path gate settings' block.


def calc_official_3h_displacements(df: pd.DataFrame, cfg) -> Dict[str, Any]:
    dists: List[float] = []
    for _, g in df.groupby(cfg.COL_SID):
        g = g.sort_values(cfg.COL_TIME)
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                dt_h = (r[cfg.COL_TIME] - prev[cfg.COL_TIME]).total_seconds() / 3600.0
                if 0 < dt_h <= 6:
                    dist = geo_distance_deg(float(r[cfg.COL_LAT]), float(r[cfg.COL_LON]),
                                            float(prev[cfg.COL_LAT]), float(prev[cfg.COL_LON]))
                    if np.isfinite(dist):
                        dists.append(float(dist * TIME_STEP_HOURS / dt_h))
            prev = r
    arr = np.asarray(dists, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "values": arr,
        "n": int(len(arr)),
        "p90": float(np.nanpercentile(arr, 90)) if len(arr) else np.nan,
        "p95": float(np.nanpercentile(arr, 95)) if len(arr) else np.nan,
        "p99": float(np.nanpercentile(arr, 99)) if len(arr) else np.nan,
        "max": float(np.nanmax(arr)) if len(arr) else np.nan,
    }


def estimate_displacement_cutoff_percentile(
    df: pd.DataFrame,
    cfg,
    user_value: Optional[float] = None,
    percentile: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Estimate the hard 3-hour displacement cutoff from official-track percentile.

    A positive user_value is treated as an explicit override. Otherwise inferred
    samples are stopped when their 3-hour displacement from the previous accepted
    point is not strictly smaller than the empirical percentile of official
    3-hour displacements.

    Default behavior:
        DISPLACEMENT_GATE_PERCENTILE = 98.0  -> official 3h displacement P98
    """
    stats = calc_official_3h_displacements(df, cfg)
    arr = np.asarray(stats.get("values", np.asarray([], dtype=np.float64)), dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size > 0:
        stats["n"] = int(arr.size)
        stats["p90"] = float(np.nanquantile(arr, 0.90))
        stats["p95"] = float(np.nanquantile(arr, 0.95))
        stats["p98"] = float(np.nanquantile(arr, 0.98))
        stats["p99"] = float(np.nanquantile(arr, 0.99))
        stats["max"] = float(np.nanmax(arr))

    stats_out = {k: v for k, v in stats.items() if k != "values"}
    p = float(DISPLACEMENT_GATE_PERCENTILE if percentile is None else percentile)
    q = p / 100.0
    disp_label = percentile_label_from_percentile(p)

    if not (0.0 < q < 1.0):
        raise ValueError(f"displacement percentile must be between 0 and 100, got {p}")

    if user_value is not None and np.isfinite(float(user_value)) and float(user_value) > 0:
        cutoff = float(user_value)
        selected_by = "user_override"
    else:
        if int(stats_out.get("n", 0)) < 30 or arr.size < 30:
            raise RuntimeError(
                f"Cannot estimate max_disp_deg from official 3h displacement {disp_label}: "
                f"valid displacement sample count={stats_out.get('n', 0)}. "
                "Please check IBTrACS data or explicitly pass --max_disp_deg."
            )
        cutoff = float(np.nanquantile(arr, q))
        if not np.isfinite(cutoff):
            raise RuntimeError(
                f"Cannot estimate max_disp_deg from official 3h displacement {disp_label}: "
                f"quantile value is {cutoff}. Please check IBTrACS data or explicitly pass --max_disp_deg."
            )
        selected_by = f"official_3h_displacement_p{int(round(p)):02d}"

    stats_out.update({
        "max_disp_deg": cutoff,
        "selected_by": selected_by,
        "definition": f"hard stop if predicted 3h displacement is not strictly smaller than official track {disp_label}",
        "displacement_gate_percentile": p,
        "displacement_gate_label": disp_label,
        "accept_rule_for_inferred": f"predicted_3h_displacement < official_{disp_label}",
    })
    print(f"Max 3h displacement cutoff selected: {cutoff:.3f} deg ({selected_by})")
    print(
        "Official 3h displacement stats: "
        f"N={stats_out.get('n')}, P90={float(stats_out.get('p90', np.nan)):.3f}, "
        f"P95={float(stats_out.get('p95', np.nan)):.3f}, "
        f"P98={float(stats_out.get('p98', np.nan)):.3f}, "
        f"P99={float(stats_out.get('p99', np.nan)):.3f}, "
        f"selected={disp_label}={cutoff:.3f}"
    )
    return cutoff, stats_out

def estimate_gate_radius(df: pd.DataFrame, cfg) -> Tuple[float, Dict[str, Any]]:
    """Use a fixed 2.0-degree core radius for VO850 feature statistics.

    This radius is not learned by the model and is not estimated from the
    displacement distribution in this version. It is fixed for reproducibility:
        vo850_core_max = max(VO850 within 2.0 deg around the center)
    """
    stats = calc_official_3h_displacements(df, cfg)
    radius = float(DEFAULT_GATE_RADIUS_DEG)
    stats_out = {k: v for k, v in stats.items() if k != "values"}
    stats_out.update({
        "gate_radius_deg": radius,
        "env_radius_deg": radius * ENV_RADIUS_FACTOR,
        "selected_by": "fixed_2.0deg",
        "clip_min_deg": SEARCH_RADIUS_MIN_DEG,
        "clip_max_deg": SEARCH_RADIUS_MAX_DEG,
        "definition": "fixed 2.0-degree core radius for VO850 core-maximum extraction and ENV_RADIUS_FACTOR-scaled radius for VWS200_850 mean extraction",
    })
    print(f"Gate/core radius fixed: {radius:.3f} deg")
    return radius, stats_out




# VO850/VWS200_850 environment-gate calibration and evaluation utilities by default.
# Official DB samples are evaluated and kept. Inferred samples are rejected if
# vo850_core_max is not strictly greater than the calibrated official percentile threshold.


def _norm_channel_name(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _get_channel_names_from_stats_or_cfg(stats: Dict[str, Any], cfg) -> List[str]:
    for key in ["channel_names", "channels", "var_names", "variables", "input_channels"]:
        val = stats.get(key) if isinstance(stats, dict) else None
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return [str(x) for x in val]
    for key in ["CHANNEL_NAMES", "CHANNELS", "VAR_NAMES", "VARIABLES", "ERA5_CHANNEL_NAMES"]:
        val = getattr(cfg, key, None)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return [str(x) for x in val]
    return []


def _find_first_channel(channel_names: List[str], candidates: List[str]) -> Optional[int]:
    norm_names = [_norm_channel_name(x) for x in channel_names]
    cand_norm = [_norm_channel_name(x) for x in candidates]
    for cand in cand_norm:
        for i, name in enumerate(norm_names):
            if name == cand:
                return i
    for cand in cand_norm:
        for i, name in enumerate(norm_names):
            if cand and cand in name:
                return i
    return None



class DynamicGateExtractor:
    """Extract VO850 and VWS200_850 diagnostics from the same ERA5 loader used by Direct-ST.

    Extracted features:
      - vo850_core_max: maximum 850-hPa relative vorticity within gate_radius_deg.
      - vws200_850_env_mean: mean 200-850-hPa vertical wind shear within
        gate_radius_deg * ENV_RADIUS_FACTOR.

    Priority for VO850:
      1. Use explicit --vo850_channel_index if supplied.
      2. Use a direct VO850 / vorticity-850 channel if channel names are available.
      3. If channel names are unavailable and the loader has the standard 9-var order,
         use index 2: [u850, v850, vo850, rh700, ttr, vws200_850, u925, v925, mslp].
      4. Fall back to computing zeta850 from U850 and V850 if both exist.

    Priority for VWS200_850:
      1. Use explicit --vws200_850_channel_index if supplied.
      2. Use a direct VWS200_850 channel if channel names are available.
      3. If channel names are unavailable and the loader has the standard 9-var order,
         use index 5.
      4. Fall back to computing sqrt((u200-u850)^2 + (v200-v850)^2) if U/V at
         200 and 850 hPa exist.

    The calibration and candidate evaluation use the same extractor, so percentile
    thresholds remain internally consistent even when the upstream ERA5 loader returns
    normalized channels.
    """

    def __init__(
        self,
        bt,
        cfg,
        stats: Dict[str, Any],
        vo850_channel_index: Optional[int] = None,
        vws200_850_channel_index: Optional[int] = None,
    ):
        self.bt = bt
        self.cfg = cfg
        self.stats = stats or {}
        self.grid = bt.GridSystem(cfg)
        self.loader = bt.Era5Loader(cfg)
        self.channel_names = _get_channel_names_from_stats_or_cfg(self.stats, cfg)
        self.n_raw_channels = self._infer_raw_channel_count()
        self.assume_standard_9var_order = (len(self.channel_names) == 0 and self.n_raw_channels >= 6)

        self.vo850_idx = int(vo850_channel_index) if vo850_channel_index is not None else self._find_vo850_channel()
        self.vws200_850_idx = (
            int(vws200_850_channel_index)
            if vws200_850_channel_index is not None
            else self._find_vws200_850_channel()
        )

        self.u850_idx, self.v850_idx = self._find_uv_channels(850)
        self.u200_idx, self.v200_idx = self._find_uv_channels(200)

        self.vo850_mode = "direct_vo850" if self.vo850_idx is not None else "computed_from_u850_v850"
        self.vws_mode = "direct_vws200_850" if self.vws200_850_idx is not None else "computed_from_uv200_uv850"

        if self.vo850_idx is None and (self.u850_idx is None or self.v850_idx is None):
            raise RuntimeError(
                "Cannot configure VO850 gate: no direct VO850/vorticity-850 channel and no U850/V850 pair were found. "
                "Please check stats['channel_names']/cfg channel names, pass --vo850_channel_index explicitly, "
                "or use the standard 9-var order. "
                f"Available channel names: {self.channel_names}; inferred raw channel count={self.n_raw_channels}"
            )

        if self.vws200_850_idx is None and (
            self.u200_idx is None or self.v200_idx is None or self.u850_idx is None or self.v850_idx is None
        ):
            raise RuntimeError(
                "Cannot configure VWS200_850 gate: no direct VWS200_850 channel and no U200/V200/U850/V850 set was found. "
                "Please check stats['channel_names']/cfg channel names, pass --vws200_850_channel_index explicitly, "
                "or use the standard 9-var order where VWS200_850 is channel index 5. "
                f"Available channel names: {self.channel_names}; inferred raw channel count={self.n_raw_channels}"
            )

        if self.vo850_idx is not None:
            print(
                f"Dynamic gate extractor: VO850 direct channel index {self.vo850_idx} "
                f"({self._channel_label(self.vo850_idx)})."
            )
        else:
            print(
                f"Dynamic gate extractor: computing VO850 from U850 index {self.u850_idx} "
                f"and V850 index {self.v850_idx}."
            )

        if self.vws200_850_idx is not None:
            print(
                f"Dynamic gate extractor: VWS200_850 direct channel index {self.vws200_850_idx} "
                f"({self._channel_label(self.vws200_850_idx)})."
            )
        else:
            print(
                f"Dynamic gate extractor: computing VWS200_850 from U200/V200 indices "
                f"{self.u200_idx}/{self.v200_idx} and U850/V850 indices {self.u850_idx}/{self.v850_idx}."
            )

    def _infer_raw_channel_count(self) -> int:
        for key in ["mean", "std"]:
            val = self.stats.get(key) if isinstance(self.stats, dict) else None
            if val is not None:
                try:
                    arr = np.asarray(val)
                    if arr.ndim >= 1 and arr.size > 0:
                        return int(arr.reshape(-1).shape[0])
                except Exception:
                    pass
        return int(len(self.channel_names))

    def _channel_label(self, idx: Optional[int]) -> str:
        if idx is None:
            return "computed"
        if 0 <= int(idx) < len(self.channel_names):
            return str(self.channel_names[int(idx)])
        if self.assume_standard_9var_order:
            standard = ["u850", "v850", "vo850", "rh700", "ttr", "vws200_850", "u925", "v925", "mslp"]
            if 0 <= int(idx) < len(standard):
                return standard[int(idx)] + "(standard_9var_assumed)"
        return "unnamed"

    def _find_vo850_channel(self) -> Optional[int]:
        candidates = [
            "vo850", "vort850", "vorticity850", "relativevorticity850", "zeta850",
            "rv850", "rvo850", "vo_850", "vort_850", "zeta_850",
        ]
        idx = _find_first_channel(self.channel_names, candidates)
        if idx is not None:
            return idx
        norm_names = [_norm_channel_name(x) for x in self.channel_names]
        for i, name in enumerate(norm_names):
            if "850" in name and ("vorticity" in name or "vort" in name or name.startswith("vo") or "zeta" in name):
                return i
        if self.assume_standard_9var_order:
            return 2
        return None

    def _find_vws200_850_channel(self) -> Optional[int]:
        candidates = [
            "vws200850", "vws200_850", "vws_200_850", "vws200to850", "vws200850hpa",
            "vws850200", "verticalwindshear200850", "verticalwindshear200_850",
            "shear200850", "shear200_850", "vws",
        ]
        idx = _find_first_channel(self.channel_names, candidates)
        if idx is not None:
            return idx
        norm_names = [_norm_channel_name(x) for x in self.channel_names]
        for i, name in enumerate(norm_names):
            if ("vws" in name or "shear" in name or "verticalwindshear" in name) and "200" in name and "850" in name:
                return i
        if self.assume_standard_9var_order:
            return 5
        return None

    def _find_uv_channels(self, level: int) -> Tuple[Optional[int], Optional[int]]:
        level_s = str(int(level))
        u_idx = _find_first_channel(
            self.channel_names,
            [f"u{level_s}", f"u_{level_s}", f"ucomponent{level_s}", f"ucomponentofwind{level_s}"],
        )
        v_idx = _find_first_channel(
            self.channel_names,
            [f"v{level_s}", f"v_{level_s}", f"vcomponent{level_s}", f"vcomponentofwind{level_s}"],
        )
        if self.assume_standard_9var_order and int(level) == 850:
            u_idx = 0 if u_idx is None else u_idx
            v_idx = 1 if v_idx is None else v_idx
        return u_idx, v_idx

    def _crop_domain(self, time_obj: pd.Timestamp) -> Optional[np.ndarray]:
        fields = self.loader.get_full_fields(pd.to_datetime(time_obj))
        if fields is None:
            return None
        dom = self.grid.crop_domain(fields)
        if dom is None:
            return None
        return np.asarray(dom, dtype=np.float32)

    def _compute_zeta_from_uv(self, dom: np.ndarray) -> np.ndarray:
        if self.u850_idx is None or self.v850_idx is None:
            raise RuntimeError("U850/V850 indices are not available for zeta computation.")
        u = np.asarray(dom[self.u850_idx], dtype=np.float64)
        v = np.asarray(dom[self.v850_idx], dtype=np.float64)

        earth_radius_m = 6_371_000.0
        lat_step_deg = float(getattr(self.cfg, "LAT_STEP", 0.25))
        lon_step_deg = float(getattr(self.cfg, "LON_STEP", 0.25))
        lat_step_m = earth_radius_m * math.radians(lat_step_deg)

        try:
            lat_map, _ = self.grid.make_latlon_maps()
            lat_arr = np.asarray(lat_map, dtype=np.float64)
            if np.nanmax(lat_arr) - np.nanmin(lat_arr) > 5.0:
                lat_row = np.nanmean(lat_arr, axis=1)
            else:
                lat_row = np.full(u.shape[0], 20.0, dtype=np.float64)
        except Exception:
            lat_row = np.full(u.shape[0], 20.0, dtype=np.float64)

        coslat = np.cos(np.deg2rad(lat_row))
        coslat = np.clip(coslat, 0.15, None)
        lon_step_m_row = earth_radius_m * math.radians(lon_step_deg) * coslat

        if abs(lat_step_m) < 1e-6 or np.nanmin(np.abs(lon_step_m_row)) < 1e-6:
            raise RuntimeError("Invalid LAT_STEP/LON_STEP for relative-vorticity computation.")

        dv_dx = np.gradient(v, axis=1) / lon_step_m_row[:, None]
        du_dy = np.gradient(u, axis=0) / lat_step_m
        return (dv_dx - du_dy).astype(np.float32)

    def _compute_vws200_850_from_uv(self, dom: np.ndarray) -> np.ndarray:
        if self.u200_idx is None or self.v200_idx is None or self.u850_idx is None or self.v850_idx is None:
            raise RuntimeError("U200/V200/U850/V850 indices are not available for VWS200_850 computation.")
        u200 = np.asarray(dom[self.u200_idx], dtype=np.float64)
        v200 = np.asarray(dom[self.v200_idx], dtype=np.float64)
        u850 = np.asarray(dom[self.u850_idx], dtype=np.float64)
        v850 = np.asarray(dom[self.v850_idx], dtype=np.float64)
        return np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2).astype(np.float32)

    def _circular_mask(self, arr_hw: np.ndarray, lat: float, lon: float, radius_deg: float) -> np.ndarray:
        height, width = arr_hw.shape
        cy, cx = self.grid.geo_to_domain_yx(float(lat), float(lon))
        yy, xx = np.ogrid[:height, :width]
        lat_step = abs(float(getattr(self.cfg, "LAT_STEP", 0.25)))
        lon_step = abs(float(getattr(self.cfg, "LON_STEP", 0.25)))
        dy_deg = (yy - float(cy)) * lat_step
        dx_deg = (xx - float(cx)) * lon_step
        return (dy_deg ** 2 + dx_deg ** 2) <= float(radius_deg) ** 2

    def extract_features(self, time_obj: pd.Timestamp, lat: float, lon: float, radius_deg: float) -> Dict[str, Any]:
        dom = self._crop_domain(time_obj)
        if dom is None:
            return {
                "valid": False,
                "reason": "missing_era5_fields",
                "vo850_valid": False,
                "vws200_850_valid": False,
                "vo850_core_max": np.nan,
                "rh700_env_mean": np.nan,
                "ttr_core_max": np.nan,
                "vws200_850_env_mean": np.nan,
            }

        vo_val = np.nan
        vws_val = np.nan
        vo_valid = False
        vws_valid = False
        reasons: List[str] = []

        try:
            if self.vo850_idx is not None:
                vo = np.asarray(dom[self.vo850_idx], dtype=np.float32)
            else:
                vo = self._compute_zeta_from_uv(dom)
            vo_mask = self._circular_mask(vo, lat, lon, radius_deg)
            vo_vals = vo[vo_mask]
            vo_vals = vo_vals[np.isfinite(vo_vals)]
            if vo_vals.size > 0:
                vo_val = float(np.nanmax(vo_vals))
                vo_valid = bool(np.isfinite(vo_val))
            if not vo_valid:
                reasons.append("invalid_vo850_core")
        except Exception as exc:
            reasons.append(f"vo850_extract_error:{type(exc).__name__}:{exc}")

        try:
            if self.vws200_850_idx is not None:
                vws = np.asarray(dom[self.vws200_850_idx], dtype=np.float32)
            else:
                vws = self._compute_vws200_850_from_uv(dom)
            vws_radius_deg = float(radius_deg) * float(ENV_RADIUS_FACTOR)
            vws_mask = self._circular_mask(vws, lat, lon, vws_radius_deg)
            vws_vals = vws[vws_mask]
            vws_vals = vws_vals[np.isfinite(vws_vals)]
            if vws_vals.size > 0:
                vws_val = float(np.nanmean(vws_vals))
                vws_valid = bool(np.isfinite(vws_val))
            if not vws_valid:
                reasons.append("invalid_vws200_850_env")
        except Exception as exc:
            reasons.append(f"vws200_850_extract_error:{type(exc).__name__}:{exc}")

        valid = bool(vo_valid and vws_valid)
        return {
            "valid": valid,
            "reason": "ok" if valid else ";".join(reasons),
            "vo850_valid": vo_valid,
            "vws200_850_valid": vws_valid,
            "vo850_core_max": vo_val,
            "rh700_env_mean": np.nan,
            "ttr_core_max": np.nan,
            "vws200_850_env_mean": vws_val,
        }


def _empty_dynamic_gate_stats(args: argparse.Namespace, reason: str) -> Dict[str, Any]:
    vo_q = float(getattr(args, "vo850_gate_percentile", VO850_GATE_PERCENTILE))
    vws_q = float(getattr(args, "vws200_850_gate_percentile", VWS200_850_GATE_PERCENTILE))
    return {
        "enabled": False,
        "variables": ENV_GATE_VARIABLES,
        "directions": ENV_GATE_DIRECTIONS,
        "reason": reason,
        "vo850_core_max": {
            "enabled": False,
            "variable": "vo850_core_max",
            "direction": "high_good",
            "percentile": vo_q,
            "percentile_label": percentile_label_from_fraction(vo_q),
        },
        "vws200_850_env_mean": {
            "enabled": False,
            "variable": "vws200_850_env_mean",
            "direction": "low_good",
            "percentile": vws_q,
            "percentile_label": percentile_label_from_fraction(vws_q),
        },
    }


def calibrate_dynamic_gate_thresholds(
    df: pd.DataFrame,
    cfg,
    extractor: DynamicGateExtractor,
    args: argparse.Namespace,
    gate_radius_deg: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Calibrate VO850 lower-bound and VWS200_850 upper-bound thresholds from official DB samples.

    Calibration samples:
      - official DB / DB_ADD status normalized to DB;
      - pre-genesis records only;
      - 0 < HOURS_TO_GENESIS <= target_history_hours;
      - within ROI;
      - seasons between cal_start_year and cal_end_year.

    Default accept rule for inferred samples:
      - vo850_core_max > official VO850 P02;
      - vws200_850_env_mean < official VWS200_850 P98.
    """
    vo_values: List[float] = []
    vws_values: List[float] = []
    n_candidates = 0
    n_missing_vo = 0
    n_missing_vws = 0
    n_out_roi = 0

    for _, group in df.groupby(cfg.COL_SID):
        group = group.sort_values(cfg.COL_TIME).reset_index(drop=True)
        if group.empty:
            continue
        genesis_time = extractor.bt.determine_genesis_time(cfg, group)
        status_upper = normalize_db_add_status(group[cfg.COL_STATUS])
        pre_db = group[(group[cfg.COL_TIME] < genesis_time) & status_upper.eq("DB")].copy()
        if pre_db.empty:
            continue
        pre_db["_HOURS_TO_GENESIS"] = (genesis_time - pre_db[cfg.COL_TIME]).dt.total_seconds() / 3600.0

        for _, row in pre_db.iterrows():
            season = int(row[cfg.COL_SEASON]) if np.isfinite(float(row[cfg.COL_SEASON])) else -9999
            if not (int(args.cal_start_year) <= season <= int(args.cal_end_year)):
                continue
            hours_to_genesis = float(row["_HOURS_TO_GENESIS"])
            if not in_time_window(hours_to_genesis, args.target_history_hours):
                continue
            lat = float(row[cfg.COL_LAT])
            lon = float(row[cfg.COL_LON])
            if not in_roi(lat, lon):
                n_out_roi += 1
                continue
            n_candidates += 1
            f = extractor.extract_features(pd.to_datetime(row[cfg.COL_TIME]), lat, lon, gate_radius_deg)

            vo = float(f.get("vo850_core_max", np.nan))
            vws = float(f.get("vws200_850_env_mean", np.nan))
            if bool(f.get("vo850_valid", False)) and np.isfinite(vo):
                vo_values.append(vo)
            else:
                n_missing_vo += 1
            if bool(f.get("vws200_850_valid", False)) and np.isfinite(vws):
                vws_values.append(vws)
            else:
                n_missing_vws += 1

    vo_arr = np.asarray(vo_values, dtype=np.float64)
    vo_arr = vo_arr[np.isfinite(vo_arr)]
    vws_arr = np.asarray(vws_values, dtype=np.float64)
    vws_arr = vws_arr[np.isfinite(vws_arr)]

    vo_q = float(getattr(args, "vo850_gate_percentile", VO850_GATE_PERCENTILE))
    vws_q = float(getattr(args, "vws200_850_gate_percentile", VWS200_850_GATE_PERCENTILE))
    if not (0.0 < vo_q < 1.0):
        raise ValueError(f"vo850_gate_percentile must be between 0 and 1, got {vo_q}")
    if not (0.0 < vws_q < 1.0):
        raise ValueError(f"vws200_850_gate_percentile must be between 0 and 1, got {vws_q}")

    if vo_arr.size < 30:
        raise RuntimeError(
            f"Cannot calibrate VO850 {percentile_label_from_fraction(vo_q)} gate: fewer than 30 valid official DB samples. "
            f"valid={vo_arr.size}, candidates={n_candidates}, missing_vo={n_missing_vo}. "
            "Check ERA5 VO850 availability, channel names, and calibration years."
        )
    if vws_arr.size < 30:
        raise RuntimeError(
            f"Cannot calibrate VWS200_850 {percentile_label_from_fraction(vws_q)} gate: fewer than 30 valid official DB samples. "
            f"valid={vws_arr.size}, candidates={n_candidates}, missing_vws={n_missing_vws}. "
            "Check ERA5 VWS200_850 availability, channel names, and calibration years."
        )

    vo_threshold = float(np.nanquantile(vo_arr, vo_q))
    vws_threshold = float(np.nanquantile(vws_arr, vws_q))

    def _stats(arr: np.ndarray, variable: str, direction: str, q: float, threshold: float, n_missing: int) -> Dict[str, Any]:
        return {
            "enabled": True,
            "variable": variable,
            "direction": direction,
            "percentile": float(q),
            "percentile_label": percentile_label_from_fraction(q),
            "threshold": float(threshold),
            "n_valid": int(arr.size),
            "n_candidates": int(n_candidates),
            "n_missing": int(n_missing),
            "n_out_roi": int(n_out_roi),
            "calibration_years": [int(args.cal_start_year), int(args.cal_end_year)],
            "target_history_hours": int(args.target_history_hours),
            "gate_radius_deg": float(gate_radius_deg),
            "env_radius_deg": float(gate_radius_deg) * float(ENV_RADIUS_FACTOR),
            "p02": float(np.nanquantile(arr, 0.02)),
            "p05": float(np.nanquantile(arr, 0.05)),
            "p10": float(np.nanquantile(arr, 0.10)),
            "p50": float(np.nanquantile(arr, 0.50)),
            "p90": float(np.nanquantile(arr, 0.90)),
            "p95": float(np.nanquantile(arr, 0.95)),
            "p99": float(np.nanquantile(arr, 0.99)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }

    vo_stats = _stats(vo_arr, "vo850_core_max", "high_good", vo_q, vo_threshold, n_missing_vo)
    vo_stats.update({
        "accept_rule_for_inferred": "vo850_core_max > threshold",
        "official_samples_removed": False,
        "inferred_samples_removed": True,
        "extractor_mode": extractor.vo850_mode,
        "vo850_channel_index": extractor.vo850_idx,
        "u850_channel_index": extractor.u850_idx,
        "v850_channel_index": extractor.v850_idx,
        "channel_names": extractor.channel_names,
    })
    vws_stats = _stats(vws_arr, "vws200_850_env_mean", "low_good", vws_q, vws_threshold, n_missing_vws)
    vws_stats.update({
        "accept_rule_for_inferred": "vws200_850_env_mean < threshold",
        "official_samples_removed": False,
        "inferred_samples_removed": True,
        "extractor_mode": extractor.vws_mode,
        "vws200_850_channel_index": extractor.vws200_850_idx,
        "u200_channel_index": extractor.u200_idx,
        "v200_channel_index": extractor.v200_idx,
        "u850_channel_index": extractor.u850_idx,
        "v850_channel_index": extractor.v850_idx,
        "channel_names": extractor.channel_names,
    })

    stats = {
        "enabled": True,
        "variables": ENV_GATE_VARIABLES,
        "directions": ENV_GATE_DIRECTIONS,
        "accept_rule_for_inferred": "vo850_core_max > vo850_threshold AND vws200_850_env_mean < vws200_850_threshold",
        "official_samples_removed": False,
        "inferred_samples_removed": True,
        "vo850_core_max": vo_stats,
        "vws200_850_env_mean": vws_stats,
    }
    thresholds = {
        "vo850_core_max": vo_threshold,
        "vws200_850_env_mean": vws_threshold,
    }

    print(
        "Dynamic gates calibrated from official DB samples: "
        f"VO850 {vo_stats['percentile_label']}={vo_threshold:.6g} | N={vo_arr.size}; "
        f"VWS200_850 {vws_stats['percentile_label']}={vws_threshold:.6g} | N={vws_arr.size}; "
        f"core_radius={gate_radius_deg:.2f} deg | env_radius={gate_radius_deg * ENV_RADIUS_FACTOR:.2f} deg"
    )
    return thresholds, stats


def evaluate_dynamic_gate(
    features: Optional[Dict[str, Any]],
    thresholds: Dict[str, float],
    gate_level: str,
    enforce_accept: bool = True,
    vo850_gate_label: str = "P02",
    vws_gate_label: str = "P98",
) -> Dict[str, Any]:
    """Evaluate the combined VO850 + VWS200_850 dynamic gate.

    Inferred samples pass only when both are true:
      - vo850_core_max > calibrated lower-bound threshold;
      - vws200_850_env_mean < calibrated upper-bound threshold.

    Official DB samples should call this with enforce_accept=False so diagnostics are
    recorded but official records are not removed.
    """
    features = features or {}
    vo = float(features.get("vo850_core_max", np.nan))
    vws = float(features.get("vws200_850_env_mean", np.nan))
    vo_threshold = float(thresholds.get("vo850_core_max", np.nan))
    vws_threshold = float(thresholds.get("vws200_850_env_mean", np.nan))

    vo_valid = bool(features.get("vo850_valid", False)) and np.isfinite(vo) and np.isfinite(vo_threshold)
    vws_valid = bool(features.get("vws200_850_valid", False)) and np.isfinite(vws) and np.isfinite(vws_threshold)

    vo_pass = bool(vo_valid and (vo > vo_threshold))
    vws_pass = bool(vws_valid and (vws < vws_threshold))
    accepted = bool(vo_pass and vws_pass) if enforce_accept else True

    flags: List[str] = []
    if not vo_valid:
        flags.append(f"vo850_invalid:{features.get('reason', 'unknown')}")
    elif vo_pass:
        flags.append(f"vo850_core_max>{vo_threshold:.6g}")
    else:
        flags.append(f"vo850_core_max<={vo850_gate_label}_threshold:{vo_threshold:.6g}")

    if not vws_valid:
        flags.append(f"vws200_850_invalid:{features.get('reason', 'unknown')}")
    elif vws_pass:
        flags.append(f"vws200_850_env_mean<{vws_threshold:.6g}")
    else:
        flags.append(f"vws200_850_env_mean>={vws_gate_label}_threshold:{vws_threshold:.6g}")

    return {
        "accepted": accepted,
        "gate_level": gate_level,
        "vo_pass": vo_pass,
        "rh_pass": np.nan,
        "ttr_pass": np.nan,
        "vws_pass": vws_pass,
        "env_pass_count": int(vo_pass) + int(vws_pass),
        "gate_flags": ";".join(flags),
    }


def append_gate_columns(rec: Dict[str, Any], features: Optional[Dict[str, Any]], gate: Optional[Dict[str, Any]],
                        gate_radius_deg: float) -> Dict[str, Any]:
    features = features or {}
    gate = gate or {}
    for k in ["vo850_core_max", "rh700_env_mean", "ttr_core_max", "vws200_850_env_mean"]:
        rec[k] = features.get(k, np.nan)
    rec["GATE_LEVEL"] = gate.get("gate_level", "Unknown")
    rec["VO_PASS"] = gate.get("vo_pass", np.nan)
    rec["RH_PASS"] = gate.get("rh_pass", np.nan)
    rec["TTR_PASS"] = gate.get("ttr_pass", np.nan)
    rec["VWS_PASS"] = gate.get("vws_pass", np.nan)
    rec["ENV_PASS_COUNT"] = gate.get("env_pass_count", np.nan)
    rec["GATE_FLAGS"] = gate.get("gate_flags", "")
    rec["GATE_RADIUS_DEG"] = gate_radius_deg
    rec["ENV_RADIUS_DEG"] = gate_radius_deg * ENV_RADIUS_FACTOR
    return rec


# =========================================================
# 4. Official records and mature-TC exclusion
# =========================================================

def load_mature_storm_db(df: pd.DataFrame, col_time: str, col_lat: str, col_lon: str, col_wind: str) -> Dict[pd.Timestamp, List[Tuple[float, float]]]:
    out: Dict[pd.Timestamp, List[Tuple[float, float]]] = {}
    if col_wind not in df.columns:
        return out
    strong = df[df[col_wind] >= MATURE_TC_MIN_WIND].dropna(subset=[col_time, col_lat, col_lon])
    for t, g in strong.groupby(col_time):
        out[pd.to_datetime(t)] = list(zip(g[col_lat].astype(float), g[col_lon].astype(float)))
    return out


def near_any_tc(lat: float, lon: float, time_obj: pd.Timestamp, tc_db: Dict[pd.Timestamp, List[Tuple[float, float]]]) -> bool:
    pts = tc_db.get(pd.to_datetime(time_obj), [])
    for tc_lat, tc_lon in pts:
        if (lat - tc_lat) ** 2 + (lon - tc_lon) ** 2 <= MATURE_TC_EXCLUSION_RADIUS_DEG ** 2:
            return True
    return False


def official_record(row: pd.Series, sid: str, genesis_time: pd.Timestamp, static: Dict[str, Any],
                    features: Optional[Dict[str, Any]], gate: Optional[Dict[str, Any]], gate_radius_deg: float) -> Dict[str, Any]:
    rec = {
        "SID": sid,
        "ISO_TIME": row["ISO_TIME"],
        "USA_LAT": row["USA_LAT"],
        "USA_LON": row["USA_LON"],
        "USA_STATUS": "DB",
        "USA_WIND": row.get("USA_WIND", np.nan),
        "DATA_SOURCE": "Official_DB_DirectST_Evaluated",
        "BACKTRACK_START_SOURCE": "OfficialDBRecord",
        "HOURS_TO_GENESIS": (genesis_time - row["ISO_TIME"]).total_seconds() / 3600.0,
        "INFER_STEP": 0,
        "STOP_REASON": "official_record",
        "MODEL_CONF": np.nan,
        "MODEL_PRIOR_SCORE": np.nan,
        "PRED_SOURCE": "official",
        "PRED_DISP_DEG": np.nan,
        "PRED_ACCEL_DEG": np.nan,
        "DIST_TO_TD_ANCHOR_DEG": geo_distance_deg(float(row["USA_LAT"]), float(row["USA_LON"]),
                                                   float(static.get("GENESIS_LAT", np.nan)), float(static.get("GENESIS_LON", np.nan))),
        "GLOBAL_LOCAL_DIST_DEG": np.nan,
        "GLOBAL_PRED_LAT": np.nan,
        "GLOBAL_PRED_LON": np.nan,
        "LOCAL_PRED_LAT": np.nan,
        "LOCAL_PRED_LON": np.nan,
        "GLOBAL_CONF": np.nan,
        "LOCAL_CONF": np.nan,
        "ANCHOR_TIME": pd.NaT,
        "ANCHOR_LAT": np.nan,
        "ANCHOR_LON": np.nan,
        "MODEL_NAME": "Official",
        "CHECKPOINT": "Official",
        "ACCEPT_RULE": "official_ibtracs_db; gate_evaluated_but_not_used_to_remove_official",
    }
    rec.update({k: v for k, v in static.items() if k not in rec})
    return append_gate_columns(rec, features, gate, gate_radius_deg)


def inferred_record(sid: str, target_time: pd.Timestamp, prev_time: pd.Timestamp,
                    prev_lat: float, prev_lon: float, pred: Dict[str, Any],
                    genesis_time: pd.Timestamp, step: int, static: Dict[str, Any],
                    model_name: str, checkpoint: str, args: argparse.Namespace,
                    features: Dict[str, Any], gate: Dict[str, Any], gate_radius_deg: float,
                    pred_disp: float, pred_accel: float, anchor_dist: float, gl_dist: float,
                    backtrack_start_source: str = "GenesisAnchor_DirectST") -> Dict[str, Any]:
    rec = {
        "SID": sid,
        "ISO_TIME": target_time,
        "USA_LAT": pred["pred_lat"],
        "USA_LON": pred["pred_lon"],
        "USA_STATUS": "DB",
        "USA_WIND": np.nan,
        "DATA_SOURCE": "Inferred_DirectST_Backtracked",
        "BACKTRACK_START_SOURCE": backtrack_start_source,
        "HOURS_TO_GENESIS": (genesis_time - target_time).total_seconds() / 3600.0,
        "INFER_STEP": step,
        "STOP_REASON": "accepted",
        "MODEL_CONF": pred.get("conf", np.nan),
        "MODEL_PRIOR_SCORE": pred.get("prior_score", np.nan),
        "PRED_SOURCE": pred.get("source", "global"),
        "PRED_DISP_DEG": pred_disp,
        "PRED_ACCEL_DEG": pred_accel,
        "DIST_TO_TD_ANCHOR_DEG": anchor_dist,
        "GLOBAL_LOCAL_DIST_DEG": gl_dist,
        "GLOBAL_PRED_LAT": pred.get("global_pred_lat", np.nan),
        "GLOBAL_PRED_LON": pred.get("global_pred_lon", np.nan),
        "LOCAL_PRED_LAT": pred.get("local_pred_lat", np.nan),
        "LOCAL_PRED_LON": pred.get("local_pred_lon", np.nan),
        "GLOBAL_CONF": pred.get("global_conf", np.nan),
        "LOCAL_CONF": pred.get("local_conf", np.nan),
        "ANCHOR_TIME": prev_time,
        "ANCHOR_LAT": prev_lat,
        "ANCHOR_LON": prev_lon,
        "MODEL_NAME": model_name,
        "CHECKPOINT": checkpoint,
        "ACCEPT_RULE": (
            f"CONF>={args.min_conf},"
            f"DISP_{percentile_label_from_percentile(getattr(args, 'displacement_gate_percentile', DISPLACEMENT_GATE_PERCENTILE))}_STRICT<{args.max_disp_deg},"
            f"VO850_CORE_MAX>{percentile_label_from_fraction(getattr(args, 'vo850_gate_percentile', VO850_GATE_PERCENTILE))},"
            f"VWS200_850_ENV_MEAN<{percentile_label_from_fraction(getattr(args, 'vws200_850_gate_percentile', VWS200_850_GATE_PERCENTILE))},"
            f"ROI,matureTC,anchorDist={args.use_anchor_distance_gate},globalLocalGate"
        ),
    }
    rec.update({k: v for k, v in static.items() if k not in rec})
    return append_gate_columns(rec, features, gate, gate_radius_deg)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "SID", "SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME",
        "ISO_TIME", "USA_LAT", "USA_LON", "USA_STATUS", "USA_WIND",
        "DATA_SOURCE", "BACKTRACK_START_SOURCE", "HOURS_TO_GENESIS", "INFER_STEP", "STOP_REASON",
        "GATE_LEVEL", "VO_PASS", "RH_PASS", "TTR_PASS", "VWS_PASS", "ENV_PASS_COUNT", "GATE_FLAGS",
        "vo850_core_max", "rh700_env_mean", "ttr_core_max", "vws200_850_env_mean",
        "MODEL_CONF", "MODEL_PRIOR_SCORE", "PRED_SOURCE", "PRED_DISP_DEG", "PRED_ACCEL_DEG",
        "DIST_TO_TD_ANCHOR_DEG", "GLOBAL_LOCAL_DIST_DEG",
        "GLOBAL_PRED_LAT", "GLOBAL_PRED_LON", "LOCAL_PRED_LAT", "LOCAL_PRED_LON", "GLOBAL_CONF", "LOCAL_CONF",
        "ANCHOR_TIME", "ANCHOR_LAT", "ANCHOR_LON",
        "MODEL_NAME", "CHECKPOINT", "ACCEPT_RULE", "GATE_RADIUS_DEG", "ENV_RADIUS_DEG",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


# =========================================================
# 5. Direct-ST one-shot path predictor
# =========================================================

class DirectSTOneShotPathPredictor:
    """Predictor adapter for the lead-selected local-canvas backtracking model.

    The reference generator expects a predictor with predict_path(...), returning
    one dict per lead step. This class builds the second model's fused input:

        [global augmented input, selected local canvas, selected local mask]

    and decodes the returned heatmap/offset predictions into the reference
    generator's expected format.
    """

    def __init__(self, bt, cfg, model: torch.nn.Module, stats: Dict[str, Any], device: str):
        self.bt = bt
        self.cfg = cfg
        self.model = model
        self.stats = stats
        self.device = device

        self.grid = bt.GridSystem(cfg)
        self.loader = bt.Era5Loader(cfg)
        self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.lat_map, self.lon_map = self.grid.make_latlon_maps()

    def _make_base_sequence(
        self,
        genesis_time: pd.Timestamp,
        genesis_lat: float,
        genesis_lon: float,
    ) -> Optional[np.ndarray]:
        """Build the global augmented ERA5 sequence [T,C_aug,H,W]."""
        x_list: List[np.ndarray] = []

        anchor_y, anchor_x = self.grid.geo_to_domain_yx(genesis_lat, genesis_lon)
        sigma_grid = self.cfg.ANCHOR_HEATMAP_SIGMA_DEG / abs(self.cfg.LON_STEP)
        anchor_map = self.bt.make_gaussian_heatmap(
            (self.cfg.DOMAIN_H, self.cfg.DOMAIN_W),
            anchor_y,
            anchor_x,
            sigma=sigma_grid,
        )[None]
        latlon_maps = np.stack([self.lat_map, self.lon_map], axis=0).astype(np.float32)

        for step_in in range(int(self.cfg.INPUT_STEPS)):
            time_i = genesis_time - timedelta(hours=int(self.cfg.TIME_STEP_HOURS) * step_in)
            fields = self.loader.get_full_fields(time_i)
            if fields is None:
                return None

            dom = self.grid.crop_domain(fields)
            if dom is None:
                return None

            dom = self.bt.normalize_domain_with_stats(dom, self.mean, self.std)

            channels = [dom]
            if getattr(self.cfg, "USE_ANCHOR_HEATMAP", True):
                channels.append(anchor_map)
            if getattr(self.cfg, "USE_LATLON_ENCODING", True):
                channels.append(latlon_maps)
            if getattr(self.cfg, "USE_LEAD_ENCODING", True):
                lead_norm = np.full(
                    (1, self.cfg.DOMAIN_H, self.cfg.DOMAIN_W),
                    float(step_in * self.cfg.TIME_STEP_HOURS) / float(self.cfg.MAX_LEAD_HOURS),
                    dtype=np.float32,
                )
                channels.append(lead_norm)

            x_list.append(np.concatenate(channels, axis=0).astype(np.float32))

        return np.stack(x_list, axis=0).astype(np.float32)

    @staticmethod
    def _input_step_lead_hour(cfg, input_step: int) -> float:
        return float(int(input_step) * int(cfg.TIME_STEP_HOURS))

    @staticmethod
    def _local_radius_deg_by_lead(cfg, lead_hour: float) -> float:
        h = float(lead_hour)
        if 0.0 <= h <= 24.0:
            return float(getattr(cfg, "LOCAL_RADIUS_0_24_DEG", 10.0))
        if 24.0 < h <= 48.0:
            return float(getattr(cfg, "LOCAL_RADIUS_24_48_DEG", 15.0))
        if 48.0 < h <= 72.0:
            return float(getattr(cfg, "LOCAL_RADIUS_48_72_DEG", 20.0))
        return float(getattr(cfg, "LOCAL_RADIUS_DEG", 20.0))

    @staticmethod
    def _make_local_canvas_chw_fallback(
        arr_chw: np.ndarray,
        center_y: float,
        center_x: float,
        radius_grid: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return one anchor-centered local area embedded in full-domain canvas."""
        channels, height, width = arr_chw.shape
        canvas = np.zeros((channels, height, width), dtype=np.float32)
        mask = np.zeros((1, height, width), dtype=np.float32)

        cy, cx = int(round(float(center_y))), int(round(float(center_x)))
        y1, y2 = cy - int(radius_grid), cy + int(radius_grid) + 1
        x1, x2 = cx - int(radius_grid), cx + int(radius_grid) + 1

        sy1, sy2 = max(0, y1), min(height, y2)
        sx1, sx2 = max(0, x1), min(width, x2)

        if sy1 < sy2 and sx1 < sx2:
            canvas[:, sy1:sy2, sx1:sx2] = arr_chw[:, sy1:sy2, sx1:sx2]
            mask[:, sy1:sy2, sx1:sx2] = 1.0

        return canvas.astype(np.float32), mask.astype(np.float32)

    def _make_lead_selected_local_canvas_sequence_fallback(
        self,
        x_seq: np.ndarray,
        center_y: float,
        center_x: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Fallback implementation of the second model's lead-selected local canvas."""
        selected_radii_deg = [
            self._local_radius_deg_by_lead(self.cfg, self._input_step_lead_hour(self.cfg, i))
            for i in range(x_seq.shape[0])
        ]

        canvas_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []

        for input_step in range(x_seq.shape[0]):
            r_deg = float(selected_radii_deg[input_step])
            r_grid = int(round(abs(r_deg / float(self.cfg.LON_STEP))))
            canvas, mask = self._make_local_canvas_chw_fallback(
                x_seq[input_step],
                center_y,
                center_x,
                r_grid,
            )
            canvas_list.append(canvas)
            mask_list.append(mask)

        local_canvas_seq = np.stack(canvas_list, axis=0).astype(np.float32)  # [T,C,H,W]
        local_mask_seq = np.stack(mask_list, axis=0).astype(np.float32)      # [T,1,H,W]
        return local_canvas_seq, local_mask_seq, selected_radii_deg

    def _get_model_expected_input_channels(self) -> Optional[int]:
        """Infer the model input channel count from its first Conv2d layer."""
        try:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    return int(module.in_channels)
        except Exception:
            pass
        return None

    def _make_lead_selected_fused_sequence(
        self,
        x_base: np.ndarray,
        genesis_lat: float,
        genesis_lon: float,
    ) -> Tuple[np.ndarray, List[float]]:
        """Build x_fused for the uploaded 27-channel lead-selected model.

        The model supplied in this conversation uses:
            C_aug = 9 ERA5 variables + 4 encodings = 13
            C_in  = [global C_aug] + [selected local C_aug] + [mask] = 27

        Therefore the local canvas is built from the same augmented tensor x_base,
        not from ERA5-only channels. A defensive check is kept so future 23-channel
        checkpoints can still be detected with a clear error message.
        """
        if not getattr(self.cfg, "USE_LOCAL_AUX_BRANCH", True):
            return x_base.astype(np.float32), []

        ay, ax = self.grid.geo_to_domain_yx(genesis_lat, genesis_lon)

        if hasattr(self.bt, "_make_lead_selected_local_canvas_sequence"):
            x_local_canvas, x_local_mask, selected_radii_deg = self.bt._make_lead_selected_local_canvas_sequence(
                x_base,
                ay,
                ax,
                self.cfg,
            )
        else:
            x_local_canvas, x_local_mask, selected_radii_deg = self._make_lead_selected_local_canvas_sequence_fallback(
                x_base,
                ay,
                ax,
            )

        # x_base         : [T,13,H,W] by default
        # x_local_canvas : [T,13,H,W], selected local canvas built from augmented channels
        # x_local_mask   : [T, 1,H,W]
        # x_fused        : [T,27,H,W] by default
        x_fused = np.concatenate([x_base, x_local_canvas, x_local_mask], axis=1).astype(np.float32)

        expected_c = self._get_model_expected_input_channels()
        if expected_c is not None and int(expected_c) != int(x_fused.shape[1]):
            raise RuntimeError(
                "Model input-channel mismatch after fused input construction: "
                f"model_expected={expected_c}, x_fused_channels={x_fused.shape[1]}. "
                "The uploaded model is expected to be the 27-channel augmented-local-canvas version. "
                "If you switch to an ERA5-only local-canvas checkpoint, use the 23-channel adapter instead."
            )

        return x_fused, [float(r) for r in selected_radii_deg]

    @staticmethod
    def _format_radius_schedule(radii: List[float]) -> str:
        if not radii:
            return ""
        # Compress repeated values, e.g. 10x9;15x8;20x8
        parts: List[str] = []
        start = 0
        while start < len(radii):
            val = radii[start]
            end = start + 1
            while end < len(radii) and abs(float(radii[end]) - float(val)) < 1e-6:
                end += 1
            parts.append(f"{float(val):.1f}degx{end - start}")
            start = end
        return ";".join(parts)

    def predict_path(
        self,
        genesis_time: pd.Timestamp,
        genesis_lat: float,
        genesis_lon: float,
    ) -> Optional[List[Dict[str, Any]]]:
        x_base = self._make_base_sequence(genesis_time, genesis_lat, genesis_lon)
        if x_base is None:
            return None

        x_fused, selected_radii_deg = self._make_lead_selected_fused_sequence(
            x_base,
            genesis_lat,
            genesis_lon,
        )

        x = torch.from_numpy(x_fused[None]).to(self.device)
        anchor = torch.tensor([[genesis_lat, genesis_lon]], dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x)

        if isinstance(out, dict):
            if "heatmap" in out and "offset" in out:
                heat, offset = out["heatmap"], out["offset"]
            elif "heat" in out and "offset" in out:
                heat, offset = out["heat"], out["offset"]
            else:
                raise RuntimeError(f"Unsupported model output dict keys: {list(out.keys())}")
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            heat, offset = out[0], out[1]
        else:
            raise RuntimeError(f"Unsupported model output type: {type(out)}")

        dec = self.bt.decode_sequence_predictions(self.cfg, self.grid, heat, offset, anchor)

        preds: List[Dict[str, Any]] = []
        radius_schedule = self._format_radius_schedule(selected_radii_deg)

        for k in range(int(self.cfg.MAX_PRED_STEPS)):
            pred_lat = float(dec["pred_lat"][0, k])
            pred_lon = float(dec["pred_lon"][0, k])
            conf = float(dec["conf"][0, k])

            prior_score = np.nan
            if isinstance(dec, dict) and "prior_score" in dec:
                prior_score = float(dec["prior_score"][0, k])

            preds.append(
                {
                    "step": int(k + 1),
                    "lead_hour": float((k + 1) * self.cfg.TIME_STEP_HOURS),
                    "pred_lat": pred_lat,
                    "pred_lon": pred_lon,
                    "conf": conf,
                    "source": "lead_selected_local_canvas",
                    "global_pred_lat": pred_lat,
                    "global_pred_lon": pred_lon,
                    "global_conf": conf,
                    "local_pred_lat": np.nan,
                    "local_pred_lon": np.nan,
                    "local_conf": np.nan,
                    "prior_score": prior_score,
                    "lead_selected_radii_deg": radius_schedule,
                    "fused_input_channels": int(x_fused.shape[1]),
                }
            )

        return preds


# =========================================================
# 6. Main generation flow
# =========================================================

def run_generation(args: argparse.Namespace) -> None:
    if args.start_year > args.end_year:
        raise ValueError("start_year must not be later than end_year.")
    if args.cal_start_year > args.cal_end_year:
        raise ValueError("cal_start_year must not be later than cal_end_year.")
    train_start, train_end = CALIBRATION_YEARS
    if args.cal_start_year < train_start or args.cal_end_year > train_end:
        raise ValueError(
            "Gate calibration must remain within the training period "
            f"{train_start}-{train_end}; validation or test years are not permitted."
        )
    if not 0.0 <= float(args.displacement_gate_percentile) <= 100.0:
        raise ValueError("displacement_gate_percentile must be between 0 and 100.")
    if not 0.0 <= float(args.vo850_gate_percentile) <= 1.0:
        raise ValueError("vo850_gate_percentile must be between 0 and 1.")
    if not 0.0 <= float(args.vws200_850_gate_percentile) <= 1.0:
        raise ValueError("vws200_850_gate_percentile must be between 0 and 1.")

    os.makedirs(args.out_dir, exist_ok=True)

    bt = load_module_from_path(args.model_script)
    cfg = bt.Config()
    cfg.IBTRACS_PATH = args.ibtracs_path
    if hasattr(cfg, "USE_ANCHOR_DISTANCE_PRIOR_DECODE"):
        cfg.USE_ANCHOR_DISTANCE_PRIOR_DECODE = not bool(args.no_anchor_prior_decode)
    if hasattr(cfg, "PRIOR_MIN_SIGMA_DEG"):
        cfg.PRIOR_MIN_SIGMA_DEG = float(args.prior_min_sigma_deg)
    if hasattr(cfg, "PRIOR_SIGMA_PER_HOUR"):
        cfg.PRIOR_SIGMA_PER_HOUR = float(args.prior_sigma_per_hour)
    if hasattr(cfg, "PRIOR_POWER"):
        cfg.PRIOR_POWER = float(args.prior_power)
    if hasattr(cfg, "LOCAL_FUSION_MAX_LEAD_HOURS"):
        cfg.LOCAL_FUSION_MAX_LEAD_HOURS = int(args.local_fusion_max_lead_hours)
    if hasattr(cfg, "LOCAL_MIN_CONF"):
        cfg.LOCAL_MIN_CONF = float(args.local_min_conf)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    print(f"Model script: {args.model_script}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"IBTrACS path: {cfg.IBTRACS_PATH}")
    print(f"Output dir  : {args.out_dir}")

    model, stats, ckpt = bt.load_model_from_checkpoint(cfg, args.checkpoint, device)
    predictor = DirectSTOneShotPathPredictor(bt, cfg, model, stats, device)

    all_tracks_df = bt.load_ibtracs(cfg)
    if cfg.COL_STATUS in all_tracks_df.columns:
        all_tracks_df[cfg.COL_STATUS] = normalize_db_add_status(all_tracks_df[cfg.COL_STATUS])

    # Keep calibration independent of the requested generation window.
    calibration_df = all_tracks_df[
        all_tracks_df[cfg.COL_SEASON].between(args.cal_start_year, args.cal_end_year)
    ].copy()
    if calibration_df.empty:
        raise RuntimeError("Gate-calibration data are empty for the configured training years.")

    df = all_tracks_df[all_tracks_df[cfg.COL_SEASON].between(args.start_year, args.end_year)].copy()
    df = df.sort_values([cfg.COL_SID, cfg.COL_TIME]).reset_index(drop=True)

    grid = bt.GridSystem(cfg)

    # Runtime labels derived from editable constants / CLI overrides.
    disp_percentile = float(getattr(args, "displacement_gate_percentile", DISPLACEMENT_GATE_PERCENTILE))
    disp_label = percentile_label_from_percentile(disp_percentile)
    vo850_gate_q = float(getattr(args, "vo850_gate_percentile", VO850_GATE_PERCENTILE))
    vo850_gate_label = percentile_label_from_fraction(vo850_gate_q)
    vws_gate_q = float(getattr(args, "vws200_850_gate_percentile", VWS200_850_GATE_PERCENTILE))
    vws_gate_label = percentile_label_from_fraction(vws_gate_q)
    env_gate_enabled = bool((not getattr(args, "disable_env_gate", False)) and (not DISABLE_ERA5_ENVIRONMENT_HARD_GATE))

    # Displacement hard-stop is calibrated from the selected official 3 h displacement percentile.
    max_disp_deg, disp_stats = estimate_displacement_cutoff_percentile(
        calibration_df, cfg, args.max_disp_deg, percentile=disp_percentile
    )
    args.max_disp_deg = float(max_disp_deg)

    # Fixed feature radius. VO850 uses this as core radius; VWS200_850 uses ENV_RADIUS_FACTOR times this radius.
    gate_radius_deg, radius_stats = estimate_gate_radius(calibration_df, cfg)
    if args.gate_radius_deg is not None and args.gate_radius_deg > 0:
        gate_radius_deg = float(args.gate_radius_deg)
        radius_stats["gate_radius_deg"] = gate_radius_deg
        radius_stats["env_radius_deg"] = gate_radius_deg * ENV_RADIUS_FACTOR
        radius_stats["selected_by"] = "user_override"
        print(f"Gate/core radius overridden by user: {gate_radius_deg:.3f} deg")

    dynamic_extractor = None
    dynamic_gate_thresholds: Dict[str, float] = {
        "vo850_core_max": float("nan"),
        "vws200_850_env_mean": float("nan"),
    }
    dynamic_gate_stats: Dict[str, Any] = _empty_dynamic_gate_stats(args, reason="environment_gate_disabled")
    if env_gate_enabled:
        dynamic_extractor = DynamicGateExtractor(
            bt,
            cfg,
            stats,
            vo850_channel_index=args.vo850_channel_index,
            vws200_850_channel_index=args.vws200_850_channel_index,
        )
        dynamic_gate_thresholds, dynamic_gate_stats = calibrate_dynamic_gate_thresholds(
            df=calibration_df,
            cfg=cfg,
            extractor=dynamic_extractor,
            args=args,
            gate_radius_deg=gate_radius_deg,
        )
    else:
        print("ERA5 environment hard gate disabled: inferred samples will not be stopped by VO850/VWS200_850.")

    vo850_threshold = float(dynamic_gate_thresholds.get("vo850_core_max", np.nan))
    vws_threshold = float(dynamic_gate_thresholds.get("vws200_850_env_mean", np.nan))
    vo850_stats = dynamic_gate_stats.get("vo850_core_max", {})
    vws_stats = dynamic_gate_stats.get("vws200_850_env_mean", {})

    thresholds = {
        "metadata": {
            "method": f"DirectST_position_with_{disp_label}_displacement_and_VO850_{vo850_gate_label}_VWS_{vws_gate_label}_dynamic_gates",
            "environment_gate_enabled": env_gate_enabled,
            "environment_gate_variables": ENV_GATE_VARIABLES,
            "environment_gate_directions": ENV_GATE_DIRECTIONS,
            "environment_gate_note": (
                f"VO850 {vo850_gate_label} lower-bound and VWS200_850 {vws_gate_label} upper-bound hard gates are enabled for inferred samples only; official DB samples are evaluated but never removed by these gates."
                if env_gate_enabled else
                "ERA5 environment hard gate disabled; official/inferred records keep NaN environment-gate diagnostics."
            ),
            "displacement_gate_percentile": float(disp_percentile),
            "displacement_gate_label": disp_label,
            "displacement_accept_rule": f"predicted_3h_displacement < official_{disp_label}",
            "dynamic_accept_rule": f"vo850_core_max > VO850_{vo850_gate_label} AND vws200_850_env_mean < VWS200_850_{vws_gate_label}",
            "gate_radius_deg": gate_radius_deg,
            "env_radius_deg": gate_radius_deg * ENV_RADIUS_FACTOR,
            "radius_stats": radius_stats,
        },
        "thresholds": dynamic_gate_stats,
    }

    gate_threshold_json = os.path.join(args.out_dir, GATE_THRESHOLD_JSON_NAME)
    with open(gate_threshold_json, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved VO850-{vo850_gate_label}-VWS-{vws_gate_label} dynamic-gate settings: {gate_threshold_json}")

    mature_db = load_mature_storm_db(df, cfg.COL_TIME, cfg.COL_LAT, cfg.COL_LON, cfg.COL_WIND)

    all_records: List[Dict[str, Any]] = []
    stop_reasons: Dict[str, int] = {}
    model_name = f"DirectST_LeadSelectedLocalCanvas_{disp_label}_VO850{vo850_gate_label}_VWS{vws_gate_label}"

    for sid, group in tqdm(df.groupby(cfg.COL_SID), desc="Direct-ST one-shot generation by SID"):
        group = group.sort_values(cfg.COL_TIME).reset_index(drop=True)
        if group.empty:
            continue
        genesis_time = bt.determine_genesis_time(cfg, group)
        status_upper = normalize_db_add_status(group[cfg.COL_STATUS])
        genesis_rows = group[group[cfg.COL_TIME] == genesis_time].copy()
        if genesis_rows.empty:
            genesis_rows = group[group[cfg.COL_TIME] >= genesis_time].sort_values(cfg.COL_TIME).head(1)
        if genesis_rows.empty:
            add_stop(stop_reasons, "missing_genesis_row")
            continue
        genesis_row = genesis_rows.iloc[0]
        genesis_lat = float(genesis_row[cfg.COL_LAT])
        genesis_lon = float(genesis_row[cfg.COL_LON])
        static = make_static_fields(genesis_row)
        static["GENESIS_LAT"] = genesis_lat
        static["GENESIS_LON"] = genesis_lon

        if not in_roi(genesis_lat, genesis_lon):
            add_stop(stop_reasons, "genesis_anchor_out_of_roi")
            continue

        pre_db_all = group[(group[cfg.COL_TIME] < genesis_time) & status_upper.eq("DB")].copy()
        if not pre_db_all.empty:
            pre_db_all["_HOURS_TO_GENESIS"] = (genesis_time - pre_db_all[cfg.COL_TIME]).dt.total_seconds() / 3600.0
        else:
            pre_db_all["_HOURS_TO_GENESIS"] = pd.Series(dtype=float)

        official_keep_by_time: Dict[pd.Timestamp, pd.Series] = {}
        if not pre_db_all.empty:
            for _, row in pre_db_all.iterrows():
                hours_to_genesis = float(row["_HOURS_TO_GENESIS"])
                if not in_time_window(hours_to_genesis, args.target_history_hours):
                    if np.isfinite(hours_to_genesis) and hours_to_genesis > args.target_history_hours:
                        add_stop(stop_reasons, "official_gt_target_history_skipped")
                    continue
                if not in_roi(float(row[cfg.COL_LAT]), float(row[cfg.COL_LON])):
                    add_stop(stop_reasons, "official_out_of_roi_skipped")
                    continue
                official_keep_by_time[pd.to_datetime(row[cfg.COL_TIME])] = row
                if env_gate_enabled and dynamic_extractor is not None:
                    f = dynamic_extractor.extract_features(
                        pd.to_datetime(row[cfg.COL_TIME]),
                        float(row[cfg.COL_LAT]),
                        float(row[cfg.COL_LON]),
                        gate_radius_deg,
                    )
                    gate = evaluate_dynamic_gate(
                        f,
                        dynamic_gate_thresholds,
                        gate_level=f"Official_VO850{vo850_gate_label}_VWS{vws_gate_label}_Evaluated",
                        enforce_accept=False,
                        vo850_gate_label=vo850_gate_label,
                        vws_gate_label=vws_gate_label,
                    )
                else:
                    f = no_environment_features()
                    gate = no_environment_gate(gate_level="Official_NoEnvGate")
                all_records.append(official_record(row, sid, genesis_time, static, f, gate, gate_radius_deg))

        genesis_season = int(genesis_row[cfg.COL_SEASON])
        if not (int(args.cal_start_year) <= genesis_season <= int(args.cal_end_year)):
            add_stop(stop_reasons, "inferred_generation_outside_training_period_skipped")
            continue

        preds = predictor.predict_path(genesis_time, genesis_lat, genesis_lon)
        if preds is None:
            add_stop(stop_reasons, "missing_era5_sequence_or_model_input")
            continue

        prev_time = genesis_time
        prev_lat = genesis_lat
        prev_lon = genesis_lon
        prev_prev_lat = np.nan
        prev_prev_lon = np.nan
        backtrack_start_source = "GenesisAnchor_DirectST"
        stopped = False

        for step in range(1, min(int(args.target_history_hours // TIME_STEP_HOURS), len(preds)) + 1):
            target_time = genesis_time - timedelta(hours=TIME_STEP_HOURS * step)
            hours_to_genesis = (genesis_time - target_time).total_seconds() / 3600.0
            if not in_time_window(hours_to_genesis, args.target_history_hours):
                add_stop(stop_reasons, "exceed_target_history")
                break

            # Official DB record is kept and used as the previous accepted point for path continuity.
            if target_time in official_keep_by_time:
                row = official_keep_by_time[target_time]
                prev_prev_lat, prev_prev_lon = prev_lat, prev_lon
                prev_time = target_time
                prev_lat = float(row[cfg.COL_LAT])
                prev_lon = float(row[cfg.COL_LON])
                continue

            pred = preds[step - 1]
            new_lat = float(pred["pred_lat"])
            new_lon = float(pred["pred_lon"])
            conf = float(pred.get("conf", np.nan))
            pred_disp = geo_distance_deg(prev_lat, prev_lon, new_lat, new_lon)
            anchor_dist = geo_distance_deg(genesis_lat, genesis_lon, new_lat, new_lon)
            gl_dist = geo_distance_deg(pred.get("global_pred_lat", np.nan), pred.get("global_pred_lon", np.nan),
                                       pred.get("local_pred_lat", np.nan), pred.get("local_pred_lon", np.nan))

            # Acceleration-like gate in degree units using previous two accepted points.
            pred_accel = np.nan
            if np.isfinite(prev_prev_lat) and np.isfinite(prev_prev_lon):
                v_prev_lat = prev_lat - prev_prev_lat
                v_prev_lon = prev_lon - prev_prev_lon
                v_new_lat = new_lat - prev_lat
                v_new_lon = new_lon - prev_lon
                pred_accel = float(np.sqrt((v_new_lat - v_prev_lat) ** 2 + (v_new_lon - v_prev_lon) ** 2))

            if not in_roi(new_lat, new_lon):
                add_stop(stop_reasons, "candidate_out_of_roi")
                stopped = True
                break

            if (not args.no_mature_tc_exclusion) and USE_MATURE_TC_EXCLUSION and near_any_tc(new_lat, new_lon, target_time, mature_db):
                add_stop(stop_reasons, "near_mature_tc")
                stopped = True
                break

            if not np.isfinite(conf) or conf < args.min_conf:
                add_stop(stop_reasons, "low_confidence")
                stopped = True
                break

            if not np.isfinite(pred_disp) or pred_disp >= args.max_disp_deg:
                add_stop(stop_reasons, "displacement_not_less_than_p98")
                stopped = True
                break

            if args.max_accel_deg > 0 and np.isfinite(pred_accel) and pred_accel > args.max_accel_deg:
                add_stop(stop_reasons, "acceleration_too_large")
                stopped = True
                break

            if args.use_anchor_distance_gate:
                max_anchor_dist = anchor_distance_limit(args, hours_to_genesis)
                if not np.isfinite(anchor_dist) or anchor_dist > max_anchor_dist:
                    add_stop(stop_reasons, "anchor_distance_too_large")
                    stopped = True
                    break

            max_gl = max_global_local_dist_for_lead(args, hours_to_genesis)
            if max_gl > 0 and np.isfinite(gl_dist) and gl_dist > max_gl:
                add_stop(stop_reasons, "global_local_inconsistent")
                stopped = True
                break

            # VO850 + VWS200_850 hard dynamic gates for inferred samples only.
            # Official samples are never deleted by these environmental/dynamic gates.
            if env_gate_enabled and dynamic_extractor is not None:
                f = dynamic_extractor.extract_features(target_time, new_lat, new_lon, gate_radius_deg)
                gate = evaluate_dynamic_gate(
                    f,
                    dynamic_gate_thresholds,
                    gate_level=f"Inferred_VO850{vo850_gate_label}_VWS{vws_gate_label}_HardGate",
                    enforce_accept=True,
                    vo850_gate_label=vo850_gate_label,
                    vws_gate_label=vws_gate_label,
                )
                if not bool(f.get("valid", False)):
                    add_stop(stop_reasons, "dynamic_features_missing")
                    stopped = True
                    break
                if not bool(gate.get("vo_pass", False)):
                    add_stop(stop_reasons, f"vo850_not_greater_than_{vo850_gate_label.lower()}")
                    stopped = True
                    break
                if not bool(gate.get("vws_pass", False)):
                    add_stop(stop_reasons, f"vws200_850_not_less_than_{vws_gate_label.lower()}")
                    stopped = True
                    break
                if not bool(gate.get("accepted", False)):
                    add_stop(stop_reasons, "dynamic_gate_rejected")
                    stopped = True
                    break
            else:
                f = no_environment_features()
                gate = no_environment_gate(gate_level="Inferred_NoEnvGate")

            all_records.append(
                inferred_record(
                    sid=sid,
                    target_time=target_time,
                    prev_time=prev_time,
                    prev_lat=prev_lat,
                    prev_lon=prev_lon,
                    pred=pred,
                    genesis_time=genesis_time,
                    step=step,
                    static=static,
                    model_name=model_name,
                    checkpoint=os.path.basename(args.checkpoint),
                    args=args,
                    features=f,
                    gate=gate,
                    gate_radius_deg=gate_radius_deg,
                    pred_disp=pred_disp,
                    pred_accel=pred_accel,
                    anchor_dist=anchor_dist,
                    gl_dist=gl_dist,
                    backtrack_start_source=backtrack_start_source,
                )
            )

            prev_prev_lat, prev_prev_lon = prev_lat, prev_lon
            prev_time = target_time
            prev_lat, prev_lon = new_lat, new_lon

        if not stopped:
            add_stop(stop_reasons, "completed_or_no_missing_slots")

    out = pd.DataFrame(all_records)
    if not out.empty:
        out["ISO_TIME"] = pd.to_datetime(out["ISO_TIME"], errors="coerce")
        out["HOURS_TO_GENESIS"] = pd.to_numeric(out["HOURS_TO_GENESIS"], errors="coerce")
        before_filter = len(out)
        out = out[out["HOURS_TO_GENESIS"].apply(lambda h: in_time_window(float(h), args.target_history_hours))].copy()
        removed_by_time_filter = before_filter - len(out)
        if removed_by_time_filter > 0:
            add_stop(stop_reasons, f"final_time_filter_removed:{removed_by_time_filter}")
        out = out.sort_values(["SID", "ISO_TIME", "DATA_SOURCE"]).reset_index(drop=True)
        out = standardize_columns(out)

    backtrack_csv = os.path.join(args.out_dir, BACKTRACK_OUTPUT_NAME)
    train_csv = os.path.join(args.out_dir, TRAIN_OUTPUT_NAME)
    stop_summary_csv = os.path.join(args.out_dir, STOP_SUMMARY_NAME)
    summary_json = os.path.join(args.out_dir, SUMMARY_JSON_NAME)

    backtrack_csv = safe_to_csv(out, backtrack_csv, "backtracked dataset")
    print(f"Saved backtracked dataset: {backtrack_csv}")

    if out.empty:
        print("No records generated.")
        append_official_beyond72_to_outputs(args)
        return

    train_df = out[out["DATA_SOURCE"].isin(["Official_DB_DirectST_Evaluated", "Inferred_DirectST_Backtracked"])].copy()
    train_df = train_df.sort_values(["SID", "ISO_TIME", "DATA_SOURCE"]).reset_index(drop=True)
    train_csv = safe_to_csv(train_df, train_csv, "train dataset")
    print(f"Saved train dataset: {train_csv} | N={len(train_df)}")

    print("\nDATA_SOURCE distribution:")
    print(out["DATA_SOURCE"].value_counts(dropna=False))
    print("\nGATE_LEVEL distribution:")
    print(out["GATE_LEVEL"].value_counts(dropna=False))
    if "BACKTRACK_START_SOURCE" in out.columns:
        print("\nBACKTRACK_START_SOURCE distribution:")
        print(out["BACKTRACK_START_SOURCE"].value_counts(dropna=False))
    if "PRED_SOURCE" in out.columns:
        print("\nPRED_SOURCE distribution:")
        print(out["PRED_SOURCE"].value_counts(dropna=False))

    stop_df = pd.Series(stop_reasons).sort_values(ascending=False).rename_axis("STOP_REASON").reset_index(name="COUNT")
    stop_summary_csv = safe_to_csv(stop_df, stop_summary_csv, "stop summary")
    print("\nStop reason distribution:")
    print(stop_df)

    summary = {
        "model_script": os.path.basename(args.model_script),
        "checkpoint": os.path.basename(args.checkpoint),
        "ibtracs_path": cfg.IBTRACS_PATH,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else None,
        "stats_channel_names": stats.get("channel_names", []),
        "years": [args.start_year, args.end_year],
        "target_history_hours": args.target_history_hours,
        "rules": {
            "min_conf": args.min_conf,
            "max_disp_deg": args.max_disp_deg,
            "max_disp_selected_by": disp_stats.get("selected_by", "unknown"),
            "max_disp_official_3h_stats": disp_stats,
            "max_accel_deg": args.max_accel_deg,
            "use_anchor_distance_gate": args.use_anchor_distance_gate,
            "anchor_dist_per_hour": args.anchor_dist_per_hour,
            "anchor_dist_min_deg": args.anchor_dist_min_deg,
            "anchor_dist_buffer_deg": args.anchor_dist_buffer_deg,
            "global_local_max_dist_0_24": args.global_local_max_dist_0_24,
            "global_local_max_dist_24_48": args.global_local_max_dist_24_48,
            "global_local_max_dist_48_72": args.global_local_max_dist_48_72,
            "environment_gate_enabled": env_gate_enabled,
            "environment_gate_variables": ENV_GATE_VARIABLES,
            "environment_gate_directions": ENV_GATE_DIRECTIONS,
            "environment_gate_note": (
                f"VO850 {vo850_gate_label} lower-bound and VWS200_850 {vws_gate_label} upper-bound hard gates enabled for inferred samples only"
                if env_gate_enabled else "ERA5 environment hard gate disabled"
            ),
            "vo850_gate_percentile": float(getattr(args, "vo850_gate_percentile", VO850_GATE_PERCENTILE)),
            "vo850_gate_label": vo850_gate_label,
            "vo850_threshold": float(vo850_threshold) if np.isfinite(vo850_threshold) else np.nan,
            "vo850_gate_stats": vo850_stats,
            "vws200_850_gate_percentile": float(getattr(args, "vws200_850_gate_percentile", VWS200_850_GATE_PERCENTILE)),
            "vws200_850_gate_label": vws_gate_label,
            "vws200_850_threshold": float(vws_threshold) if np.isfinite(vws_threshold) else np.nan,
            "vws200_850_gate_stats": vws_stats,
            "dynamic_gate_thresholds": dynamic_gate_thresholds,
            "dynamic_gate_stats": dynamic_gate_stats,
            "displacement_gate_percentile": float(disp_percentile),
            "gate_radius_deg": gate_radius_deg,
            "env_radius_deg": gate_radius_deg * ENV_RADIUS_FACTOR,
            "use_mature_tc_exclusion": bool(USE_MATURE_TC_EXCLUSION and not args.no_mature_tc_exclusion),
            "strict_hours_to_genesis_filter": f"0 < HOURS_TO_GENESIS <= {args.target_history_hours}",
            "official_records_not_removed_by_environment_gate": True,
            "inferred_records_removed_if_vo850_not_greater_than_threshold": bool(env_gate_enabled),
            "inferred_records_removed_if_vws200_850_not_less_than_threshold": bool(env_gate_enabled),
            "inferred_records_removed_if_dynamic_gate_not_satisfied": bool(env_gate_enabled),
            "inferred_records_stop_at_first_non_environment_gate_failure": True,
        },
        "output_files": {
            "backtrack_csv": backtrack_csv,
            "train_csv": train_csv,
            "stop_summary_csv": stop_summary_csv,
            "summary_json": summary_json,
            "dynamic_gate_settings_json": gate_threshold_json,
        },
        "counts": {
            "all_records": int(len(out)),
            "official_records": int((out["DATA_SOURCE"] == "Official_DB_DirectST_Evaluated").sum()),
            "inferred_records": int((out["DATA_SOURCE"] == "Inferred_DirectST_Backtracked").sum()),
            "train_records": int(len(train_df)),
        },
        "data_source_distribution": out["DATA_SOURCE"].value_counts(dropna=False).to_dict(),
        "gate_level_distribution": out["GATE_LEVEL"].value_counts(dropna=False).to_dict(),
        "backtrack_start_source_distribution": out["BACKTRACK_START_SOURCE"].value_counts(dropna=False).to_dict() if "BACKTRACK_START_SOURCE" in out.columns else {},
        "pred_source_distribution": out["PRED_SOURCE"].value_counts(dropna=False).to_dict() if "PRED_SOURCE" in out.columns else {},
        "stop_reasons": stop_reasons,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved summary JSON: {summary_json}")

    append_official_beyond72_to_outputs(args)




# =========================================================
# 7. Official >72 h sample preservation for target detection training
# =========================================================

def _load_python_module_from_path(path: str, module_name: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"model_script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _get_name_value(row: pd.Series) -> str:
    for c in ["NAME", "USA_NAME", "STORM_NAME"]:
        if c in row.index and pd.notna(row.get(c)):
            return str(row.get(c))
    return ""


def _fallback_determine_genesis_time(cfg, group: pd.DataFrame) -> pd.Timestamp:
    status = group[getattr(cfg, "COL_STATUS", "USA_STATUS")].astype(str).str.upper()
    wind_col = getattr(cfg, "COL_WIND", "USA_WIND")
    if wind_col in group.columns:
        wind = pd.to_numeric(group[wind_col], errors="coerce")
    else:
        wind = pd.Series(np.nan, index=group.index)

    cond_td = status.eq("TD")
    cond_other = (~status.isin(["DB", "DB_ADD", "TD"])) & (wind >= 25)
    cand = group[cond_td | cond_other]
    if not cand.empty:
        return pd.to_datetime(cand[getattr(cfg, "COL_TIME", "ISO_TIME")].min())
    return pd.to_datetime(group[getattr(cfg, "COL_TIME", "ISO_TIME")].min())


def _build_official_beyond72_records(
    bt,
    cfg,
    max_hours: Optional[float] = OFFICIAL_BEYOND72_MAX_HOURS,
) -> pd.DataFrame:
    """Create official DB/DB_add rows with HOURS_TO_GENESIS > 72 h.

    These rows are intended for the target-detection training CSV. They are not
    generated by the backtracking model; they are official pre-genesis records.
    """
    if not hasattr(bt, "load_ibtracs"):
        raise RuntimeError("The model script does not expose load_ibtracs(cfg).")

    df = bt.load_ibtracs(cfg)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    col_sid = getattr(cfg, "COL_SID", "SID")
    col_time = getattr(cfg, "COL_TIME", "ISO_TIME")
    col_lat = getattr(cfg, "COL_LAT", "USA_LAT")
    col_lon = getattr(cfg, "COL_LON", "USA_LON")
    col_wind = getattr(cfg, "COL_WIND", "USA_WIND")
    col_season = getattr(cfg, "COL_SEASON", "SEASON")
    col_status = getattr(cfg, "COL_STATUS", "USA_STATUS")

    df[col_time] = pd.to_datetime(df[col_time], format="mixed", errors="coerce")
    df = df.dropna(subset=[col_sid, col_time, col_lat, col_lon, col_season]).copy()

    if hasattr(cfg, "ALL_YEARS") and cfg.ALL_YEARS is not None:
        y0, y1 = cfg.ALL_YEARS
        df = df[pd.to_numeric(df[col_season], errors="coerce").between(y0, y1)].copy()

    rows: List[Dict[str, Any]] = []

    for sid, g in df.groupby(col_sid):
        g = g.sort_values(col_time).reset_index(drop=True)
        if g.empty:
            continue

        if hasattr(bt, "determine_genesis_time"):
            genesis_time = pd.to_datetime(bt.determine_genesis_time(cfg, g))
        else:
            genesis_time = _fallback_determine_genesis_time(cfg, g)

        genesis_rows = g[g[col_time] == genesis_time]
        if genesis_rows.empty:
            continue
        genesis_row = genesis_rows.iloc[0]

        genesis_lat = _safe_float(genesis_row.get(col_lat))
        genesis_lon = _safe_float(genesis_row.get(col_lon))
        season = int(_safe_float(genesis_row.get(col_season), -1))
        name = _get_name_value(genesis_row)

        # Use the model script's in_domain function if available.
        if hasattr(bt, "in_domain"):
            if not bt.in_domain(cfg, genesis_lat, genesis_lon):
                continue

        status = g[col_status].astype(str).str.upper() if col_status in g.columns else pd.Series("", index=g.index)
        official_db = g[(g[col_time] < genesis_time) & status.str.startswith("DB")].copy()
        if official_db.empty:
            continue

        for _, r in official_db.iterrows():
            sample_time = pd.to_datetime(r[col_time])
            hours_to_genesis = (genesis_time - sample_time).total_seconds() / 3600.0
            if not np.isfinite(hours_to_genesis) or hours_to_genesis <= 72.0:
                continue
            if max_hours is not None and hours_to_genesis > float(max_hours):
                continue

            lat = _safe_float(r.get(col_lat))
            lon = _safe_float(r.get(col_lon))
            if hasattr(bt, "in_domain"):
                if not bt.in_domain(cfg, lat, lon):
                    continue

            split = "ignore"
            if hasattr(cfg, "TRAIN_YEARS") and cfg.TRAIN_YEARS[0] <= season <= cfg.TRAIN_YEARS[1]:
                split = "train"
            elif hasattr(cfg, "VAL_YEARS") and cfg.VAL_YEARS[0] <= season <= cfg.VAL_YEARS[1]:
                split = "val"

            rows.append(
                {
                    "SID": str(sid),
                    "NAME": name,
                    "SEASON": season,
                    "SPLIT": split,
                    "ISO_TIME": sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "USA_LAT": lat,
                    "USA_LON": lon,
                    "USA_STATUS": str(r.get(col_status, "DB")).upper(),
                    "USA_WIND": _safe_float(r.get(col_wind)),
                    "HOURS_TO_GENESIS": float(hours_to_genesis),
                    "LEAD_HOUR": float(hours_to_genesis),
                    "GENESIS_TIME": genesis_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "GENESIS_LAT": genesis_lat,
                    "GENESIS_LON": genesis_lon,
                    "GENESIS_STATUS": str(genesis_row.get(col_status, "")).upper(),
                    "GENESIS_WIND": _safe_float(genesis_row.get(col_wind)),
                    "DATA_SOURCE": "Official_Beyond72_DB",
                    "SOURCE_TYPE": "Official",
                    "POINT_TYPE": "official_beyond72",
                    "MODEL_NAME": "Official_Beyond72_DB",
                    "IS_OFFICIAL": 1,
                    "IS_INFERRED": 0,
                    "IS_BEYOND72_OFFICIAL": 1,
                    "CONF": np.nan,
                    "PRED_LAT": np.nan,
                    "PRED_LON": np.nan,
                    "ERROR_DEG": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    return out


def _ensure_detection_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DETECTION_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    # Keep common aliases consistent if they exist.
    if "LAT" in out.columns and out["USA_LAT"].isna().all():
        out["USA_LAT"] = out["LAT"]
    if "LON" in out.columns and out["USA_LON"].isna().all():
        out["USA_LON"] = out["LON"]
    if "TIME" in out.columns and out["ISO_TIME"].isna().all():
        out["ISO_TIME"] = out["TIME"]

    return out


def _append_records_to_csv(csv_path: str, records_df: pd.DataFrame) -> pd.DataFrame:
    """Append official >72 h records to one output CSV and de-duplicate by SID/time."""
    if records_df is None or records_df.empty:
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path, low_memory=False)
        return pd.DataFrame()

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        base = pd.read_csv(csv_path, low_memory=False)
    else:
        base = pd.DataFrame()

    base = _ensure_detection_columns(base)
    add = _ensure_detection_columns(records_df)

    # Union columns without losing existing generator-specific fields.
    all_cols = list(base.columns)
    for c in add.columns:
        if c not in all_cols:
            all_cols.append(c)
    base = base.reindex(columns=all_cols)
    add = add.reindex(columns=all_cols)

    merged = pd.concat([base, add], ignore_index=True)

    # Normalize de-duplication keys.
    merged["_DEDUP_TIME"] = pd.to_datetime(merged["ISO_TIME"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    merged["_DEDUP_SID"] = merged["SID"].astype(str)
    merged["_OFFICIAL_BEYOND72_PRIORITY"] = np.where(
        merged.get("DATA_SOURCE", "").astype(str).str.startswith("Official_Beyond72"),
        0,
        1,
    )

    merged = merged.sort_values(["_DEDUP_SID", "_DEDUP_TIME", "_OFFICIAL_BEYOND72_PRIORITY"], na_position="last")
    merged = merged.drop_duplicates(subset=["_DEDUP_SID", "_DEDUP_TIME"], keep="first")
    merged = merged.drop(columns=["_DEDUP_TIME", "_DEDUP_SID", "_OFFICIAL_BEYOND72_PRIORITY"], errors="ignore")

    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return merged


def _report_official_beyond72(csv_path: str, label: str) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        print(f"[Official >72h check] {label}: file not found -> {csv_path}")
        return {"path": csv_path, "exists": False, "rows": 0}

    df = pd.read_csv(csv_path, low_memory=False)
    if "HOURS_TO_GENESIS" not in df.columns or "DATA_SOURCE" not in df.columns:
        print(f"[Official >72h check] {label}: missing HOURS_TO_GENESIS or DATA_SOURCE")
        return {"path": csv_path, "exists": True, "rows": 0}

    hours = pd.to_numeric(df["HOURS_TO_GENESIS"], errors="coerce")
    ds = df["DATA_SOURCE"].astype(str)
    mask = ds.str.startswith("Official") & (hours > 72.0)

    sub = df[mask].copy()
    rows = int(len(sub))
    unique_sid = int(sub["SID"].astype(str).nunique()) if "SID" in sub.columns and rows > 0 else 0
    min_h = float(pd.to_numeric(sub["HOURS_TO_GENESIS"], errors="coerce").min()) if rows > 0 else np.nan
    max_h = float(pd.to_numeric(sub["HOURS_TO_GENESIS"], errors="coerce").max()) if rows > 0 else np.nan

    print(
        f"[Official >72h check] {label}: rows={rows} | unique_sid={unique_sid} "
        f"| min_h={min_h if np.isfinite(min_h) else 'nan'} | max_h={max_h if np.isfinite(max_h) else 'nan'}"
    )

    check_csv = os.path.join(os.path.dirname(csv_path), f"official_beyond72_check_{label}.csv")
    sub.to_csv(check_csv, index=False, encoding="utf-8-sig")
    print(f"[Official >72h check] saved detail: {check_csv}")

    return {
        "path": csv_path,
        "exists": True,
        "rows": rows,
        "unique_sid": unique_sid,
        "min_hours": min_h,
        "max_hours": max_h,
        "detail_csv": check_csv,
    }


def append_official_beyond72_to_outputs(args) -> None:
    """Append official >72 h samples to generator outputs and verify them."""
    if not KEEP_OFFICIAL_BEYOND72:
        print("[Official >72h] KEEP_OFFICIAL_BEYOND72=False, skip.")
        return

    out_dir = getattr(args, "out_dir", DEFAULT_OUT_DIR)
    model_script = getattr(args, "model_script", DEFAULT_MODEL_SCRIPT)

    try:
        bt = _load_python_module_from_path(model_script, "bt_lead_selected_for_beyond72_append")
        cfg = bt.Config()
        if hasattr(args, "ibtracs_path"):
            cfg.IBTRACS_PATH = args.ibtracs_path
        if hasattr(cfg, "__post_init__"):
            cfg.__post_init__()
        if hasattr(args, "ibtracs_path"):
            cfg.IBTRACS_PATH = args.ibtracs_path
        records = _build_official_beyond72_records(
            bt=bt,
            cfg=cfg,
            max_hours=OFFICIAL_BEYOND72_MAX_HOURS,
        )
    except Exception as exc:
        print(f"[Official >72h] Failed to build official beyond-72 h records: {repr(exc)}")
        return

    print(
        "[Official >72h] built official records: "
        f"rows={len(records)} | unique_sid={records['SID'].nunique() if not records.empty and 'SID' in records.columns else 0} "
        f"| max_hours={OFFICIAL_BEYOND72_MAX_HOURS}"
    )

    train_csv = os.path.join(out_dir, TRAIN_OUTPUT_NAME)
    backtrack_csv = os.path.join(out_dir, BACKTRACK_OUTPUT_NAME)

    merged_train = _append_records_to_csv(train_csv, records)
    merged_backtrack = _append_records_to_csv(backtrack_csv, records)

    summary = {
        "keep_official_beyond72": KEEP_OFFICIAL_BEYOND72,
        "displacement_gate_percentile": float(DISPLACEMENT_GATE_PERCENTILE),
        "official_beyond72_max_hours": OFFICIAL_BEYOND72_MAX_HOURS,
        "generated_official_beyond72_rows": int(len(records)),
        "train_csv": _report_official_beyond72(train_csv, "train"),
        "backtrack_csv": _report_official_beyond72(backtrack_csv, "backtrack"),
        "required_detection_columns": DETECTION_REQUIRED_COLUMNS,
        "train_csv_columns_ok": all(c in merged_train.columns for c in DETECTION_REQUIRED_COLUMNS) if not merged_train.empty else False,
        "backtrack_csv_columns_ok": all(c in merged_backtrack.columns for c in DETECTION_REQUIRED_COLUMNS) if not merged_backtrack.empty else False,
    }

    summary_path = os.path.join(out_dir, "Official_Beyond72_TargetDetection_Check.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Official >72h] summary saved: {summary_path}")

    if summary["train_csv"]["rows"] <= 0:
        print(
            "[Official >72h][WARNING] Training CSV still contains no official >72 h samples. "
            "Please check whether the source IBTrACS/DB_add CSV has DB/DB_add records beyond 72 h."
        )



# =========================================================
# 8. CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_script", type=str, default=DEFAULT_MODEL_SCRIPT, help="Direct-ST model code path")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="trained model checkpoint")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="output directory")
    p.add_argument("--ibtracs_path", type=str, default=DEFAULT_IBTRACS_PATH, help="DB_ADD IBTrACS CSV path")
    p.add_argument("--start_year", type=int, default=BACKTRACK_YEARS[0])
    p.add_argument("--end_year", type=int, default=BACKTRACK_YEARS[1])
    p.add_argument("--cal_start_year", type=int, default=CALIBRATION_YEARS[0])
    p.add_argument("--cal_end_year", type=int, default=CALIBRATION_YEARS[1])
    p.add_argument("--target_history_hours", type=int, default=TARGET_HISTORY_HOURS)

    p.add_argument("--min_conf", type=float, default=DEFAULT_MIN_CONF, help="minimum selected heatmap confidence")
    p.add_argument(
        "--max_disp_deg", type=float, default=DEFAULT_MAX_DISP_DEG,
        help="maximum accepted 3 h displacement; default None means official 3h displacement P98"
    )
    p.add_argument(
        "--displacement_gate_percentile", type=float, default=DISPLACEMENT_GATE_PERCENTILE,
        help="official-track 3 h displacement percentile used when --max_disp_deg is not given; default 98 = P98"
    )
    p.add_argument("--max_accel_deg", type=float, default=DEFAULT_MAX_ACCEL_DEG, help="maximum acceleration-like jump; 0 disables")
    p.add_argument("--use_anchor_distance_gate", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_ANCHOR_DISTANCE_GATE)
    p.add_argument("--anchor_dist_per_hour", type=float, default=DEFAULT_ANCHOR_DIST_PER_HOUR)
    p.add_argument("--anchor_dist_min_deg", type=float, default=DEFAULT_ANCHOR_DIST_MIN_DEG)
    p.add_argument("--anchor_dist_buffer_deg", type=float, default=DEFAULT_ANCHOR_DIST_BUFFER_DEG)
    p.add_argument("--global_local_max_dist_0_24", type=float, default=DEFAULT_GLOBAL_LOCAL_MAX_DIST_0_24)
    p.add_argument("--global_local_max_dist_24_48", type=float, default=DEFAULT_GLOBAL_LOCAL_MAX_DIST_24_48)
    p.add_argument("--global_local_max_dist_48_72", type=float, default=DEFAULT_GLOBAL_LOCAL_MAX_DIST_48_72)

    p.add_argument("--gate_radius_deg", type=float, default=DEFAULT_GATE_RADIUS_DEG, help="fixed VO850 core radius in degrees; VWS uses radius * ENV_RADIUS_FACTOR; default 2.0")
    p.add_argument(
        "--disable_env_gate", action="store_true",
        help="disable ERA5 environmental hard gate for inferred samples; features will be filled as NaN"
    )
    p.add_argument("--vo850_gate_percentile", type=float, default=VO850_GATE_PERCENTILE, help="training-period official DB quantile for the VO850 lower-bound gate; default 0.02 = P02")
    p.add_argument("--vws200_850_gate_percentile", type=float, default=VWS200_850_GATE_PERCENTILE, help="training-period official DB quantile for the VWS200_850 upper-bound gate; default 0.98 = P98")
    p.add_argument("--vo850_channel_index", type=int, default=None, help="optional direct VO850 channel index; overrides automatic channel-name matching")
    p.add_argument("--vws200_850_channel_index", type=int, default=None, help="optional direct VWS200_850 channel index; overrides automatic channel-name matching")

    p.add_argument("--no_anchor_prior_decode", action="store_true", help="disable model decoding anchor-distance prior")
    p.add_argument("--prior_min_sigma_deg", type=float, default=2.0)
    p.add_argument("--prior_sigma_per_hour", type=float, default=0.35)
    p.add_argument("--prior_power", type=float, default=1.0)
    p.add_argument("--local_fusion_max_lead_hours", type=int, default=72, help="allow local branch fusion up to this lead if model supports it")
    p.add_argument("--local_min_conf", type=float, default=0.10)

    p.add_argument("--no_mature_tc_exclusion", action="store_true", help="disable mature TC proximity exclusion")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args()


if __name__ == "__main__":
    run_generation(parse_args())

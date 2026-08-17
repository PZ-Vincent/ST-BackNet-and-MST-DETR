# ST-BackNet and MST-DETR

Implementation of a two-stage tropical-disturbance detection framework. This
code release contains ST-BackNet, MST-DETR and five MST-DETR ablation models.

## Included code

```text
src/stage1/
  train_st_backnet.py
  generate_backtracked_labels.py

src/stage2/
  train_mst_detr.py

src/ablations/
  ablation_no_temporal_context.py
  ablation_no_spatial_gate.py
  ablation_no_era5.py
  ablation_no_gridsat.py
  ablation_no_time_encoding.py
```

### Stage 1: ST-BackNet

`train_st_backnet.py` trains the spatiotemporal backtracking network used to
estimate pre-genesis disturbance centres from ERA5 sequences.

`generate_backtracked_labels.py` applies a trained ST-BackNet checkpoint and
uses displacement, confidence and environmental constraints to generate the
backtracked training labels used by the second stage.

### Stage 2: MST-DETR

`train_mst_detr.py` implements multimodal tropical-disturbance detection using
GridSat and ERA5 inputs, gated feature fusion, 3D-CNN spatiotemporal context and
DETR-based detection heads.

### Ablation models

- `ablation_no_temporal_context.py` removes the 3D-CNN temporal-context path.
- `ablation_no_spatial_gate.py` replaces gated multimodal fusion with plain
  concatenation and convolution.
- `ablation_no_era5.py` retains only the GridSat branch.
- `ablation_no_gridsat.py` retains only the ERA5 branch.
- `ablation_no_time_encoding.py` removes cyclic month/hour encoding from the
  lead-time head.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

PyTorch and TorchVision builds should be selected for the CUDA version of the
target system. The DETR backbone is loaded through Torch Hub from a fixed commit
of the official `facebookresearch/detr` repository. Internet access is required
only when the pinned source and pretrained weights are not already cached.

## Configuration

Set `TC_DATA_ROOT` to the local input root and `TC_OUTPUT_ROOT` to the desired
output location. If `TC_OUTPUT_ROOT` is unset, scripts write to the repository's
local `outputs` directory. Dataset splits, channel selections, thresholds,
optimisation settings and random seeds are defined in each script's `Config`
class.

## Evaluation protocol

The default temporal split is 2000-2017 for training, 2018-2019 for validation
and 2020-2023 for testing. Environmental and displacement thresholds used to
generate inferred labels are calibrated exclusively on the training period,
and inferred labels are generated only for that period.

MST-DETR checkpoints are ranked using validation CSI at the predefined
confidence threshold of 0.5. The test split is constructed only after
validation-based checkpoint selection and is evaluated once for final
reporting. Test metrics are not used for checkpoint selection and are not
embedded in checkpoint filenames.

## Usage

Train ST-BackNet:

```bash
python src/stage1/train_st_backnet.py --mode all
```

Generate backtracked labels:

```bash
python src/stage1/generate_backtracked_labels.py \
  --checkpoint PATH_TO_ST_BACKNET_CHECKPOINT
```

Train and evaluate MST-DETR:

```bash
python src/stage2/train_mst_detr.py
```

Run an ablation model by executing the corresponding script in
`src/ablations/`. The released configurations use random seed 42.

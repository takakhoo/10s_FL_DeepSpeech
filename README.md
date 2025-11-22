# Recon Experiment Log

This file tracks every concrete action taken inside `/scratch2/f004h1v/recon` and
serves as the runbook for repeating both the short-form baselines and the new
10-second dummy-model reconstruction experiments. Update the log table whenever
you run code, change a file, or move outputs so we can retrace mistakes later.

## Repository Map
- `src/main.py` – CLI entry point, dataset loading, and logging bootstrap.
- `src/optimize.py` – first/zero-order loops, gradient matching logic, grid
  chunking of the dummy input (`x_param`).
- `src/data/librisubset.py` – HDF5 reader, batching, and LibriSpeech transforms.
- `modules/deepspeech/src/*` – vendor DeepSpeech models, preprocessors, and the
  alphabet definitions used for decoding reconstructed transcripts.
- `run.bash` – historical DS1 sweep (kept for reference).
- `tmp_make_long_dataset.py` (repo root) – helper that extracts a single 10 s
  LibriSpeech sample into HDF5 for long-form tests.
- `logging/` – auto-created experiment directories with logs, loss curves, and
  tensors (one folder per duration bucket + config signature).
- `notebooks/` – scratch analyses; useful for visualizing reconstructions.

## Environment & One-Time Setup
1. `cd /scratch2/f004h1v/recon`
2. `conda env create -f environment.yml && conda activate recon`  
   (or `pip install -r requirements.txt` inside the preferred environment)
3. Ensure the DeepSpeech submodule is importable:
   `export PYTHONPATH=$PWD/modules/deepspeech/src:$PYTHONPATH`
4. (Optional) symlink any additional datasets into `datasets/` if the shared
   ones are missing.

## Baseline Short-Form Run (<=4 s, real data, DS1)
Minh's original experiment used DeepSpeech1 with MFCC inputs. Run this first to
make sure the reconstruction loop/plots still line up with his figures.

```bash
python src/main.py \
  --model_name ds1 \
  --batch_start_idx 0 \
  --batch_end_idx 1 \
  --min_duration_ms 0 \
  --max_duration_ms 1000 \
  --learning_rate 0.5 \
  --max_iterations 2000 \
  --context_frames 6 \
  --dropout_prob 0.0
```

Outputs are written to
`logging/0s-1s/DS1_batchstart_0_batch_end_1_...`. In the latest run (16:37 EST)
the feature-space MAE converged to ≈2.6 with the expected loss curves and
spectrograms saved under `figures/`.

## Long-Form Dummy-Model Experiment (10 s, DS1)
We now chunk the **dummy input** (`x_param`) while feeding the full 10-second
gradients from the real audio through the DS1 model. Grid mode keeps only one
segment of the dummy input active at a time.

### 1. Prepare the dataset
```bash
cd /scratch2/f004h1v
python tmp_make_long_dataset.py
```
This creates `local_datasets/librispeech_long_10s/dataset_item_0.h5`.
Verify the metadata with `h5ls dataset_item_0.h5`.

### 2. Launch reconstruction with grid-based chunking
```bash
cd /scratch2/f004h1v/recon
python src/main.py \
  --model_name ds1 \
  --dataset_path ../local_datasets/librispeech_long_10s \
  --batch_start_idx 0 \
  --batch_end_idx 1 \
  --min_duration_ms 0 \
  --max_duration_ms 13000 \
  --learning_rate 0.05 \
  --max_iterations 4000 \
  --use_grid_optimization \
  --grid_size 300 \
  --grid_overlap 150 \
  --context_frames 6 \
  --dropout_prob 0.0
```

- `grid_size`/`grid_overlap` define the dummy-model chunk size and overlap. For
  a 10 s clip, 300/150 keeps the optimizer focused on ~3 s spans.
- `set_up_dir` now auto-creates `logging/0s-13s/…` for any duration bucket past
  4 s so long-form artifacts do not overwrite the short-run folders.
- Use `--checkpoint_path /path/to/pretrained.pt` if you need Minh's weights
  instead of the random init.
- Adam with `learning_rate=0.05` avoids the NaNs that appeared at 0.1; this
  matches Minh’s DS1 loss (`batched_ctc_v2`) but adds per-step normalization to
  keep gradients finite for 10 s clips.

### 3. Collect metrics & spectrograms
- All console + optimizer stats land in
  `logging/0s-10s/.../experiment.log`.  
  `grep 'Iter' experiment.log` is a quick way to extract MAE and gradient norms.
- Intermediate spectrograms + loss curves live in
  `logging/.../figures`.
- The final tensors are stored as `sampleidx_*_x_param_last.pt` with the
  following keys: `x_param`, `inputs`, `targets`, `transcript`, `time`.

Feature-space MAE and SNR can be computed offline:

```bash
python - <<'PY'
import torch, math
payload = torch.load('logging/0s-10s/.../sampleidx_0_x_param_last.pt')
x_hat, x_gt = payload['x_param'], payload['inputs']
mae = torch.mean(torch.abs(x_hat - x_gt)).item()
snr = 20 * math.log10(x_gt.norm().item() / (x_gt - x_hat).norm().item())
print(f"MAE={mae:.6f}, SNR={snr:.2f} dB")
PY
```

To derive WER:
1. Convert the reconstructed spectrogram back to a waveform (Griffin-Lim or
   another phase-recovery step via `librosa.istft` – see
   `notebooks/008_ds2_reconstruct.ipynb` for a template).
2. Feed the waveform through DeepSpeech's decoder
   (`modules/deepspeech/src/deepspeech/run.py`) and compare the predicted text
   with `payload['transcript']`.

Document exact metric values in this README once each run finishes. The present
10 s runs still need hyper-parameter tuning: with only 40 warm-up iterations
the grid optimizer reports `nan` gradients (see `logging/0s-13s/...`). Increase
iterations gradually once the NaNs are resolved.

### 4. Grid diagnostics & checkpoints
- Every 100 iterations **per grid** now writes plots whose filenames encode the
  sample index, grid index, and iteration
  (`sampleidx_0_grid_grid1_iter200_loss_curves_0.png`, etc.).
- After each grid completes we save
  `sampleidx_{i}_grid{grid_idx}_checkpoint.pt` inside the experiment folder.
  On restart, the optimizer automatically reloads any existing grid checkpoints
  and skips grids that already converged, so deleting a checkpoint is the way to
  force a redo.
- Run long jobs via `timeout 4h python ...` (or nohup/tmux) so all four grids
  finish without being killed by the shell; lisplab-1 has idle RTX 6000 Ada
  GPUs confirmed via `nvidia-smi`.

### 5. Observations from the 22 Nov 10 s run
- Config: DS1, `lr=0.05`, `max_iterations=4000`, `grid_size=300`, `grid_overlap=150`, lisplab‑1 GPU0. Logs + artifacts live in  
  `logging/0s-13s/DS1_batchstart_0_batch_end_1_init_uniform_opt_Adam_lr_0.05_reg_None_regw_0.0_top-grad-perc_1.0_cpt_None/`.
- Runtime: 4,656 s (`long_run.out` records every grid iteration).
- Metrics (from `analysis/metrics.json`):
  - Global MAE = 10.73 (per-grid MAE ranges 9.79→7.93 in the tail where the clip fades).
  - Global SNR = ‑1.80 dB.
  - Decoder hypothesis is nonsense and WER=1.0 because `generate_metrics.py` currently instantiates a **randomly initialized** DS1 when no checkpoint is supplied.
  - The reconstructed waveform created by `librosa.feature.inverse.mfcc_to_audio` sounds harsh because we never undo MFCC mean/std normalization nor the ±6 context frames.
- Takeaways / mistakes:
  1. **CTC mismatch.** `src/main.py` was temporarily switched to PyTorch’s `CTCLoss` for gradient generation, but the optimizer still uses Minh’s `batched_ctc_v2` inside `meta_loss`. This means the dummy is matching gradients from two different losses, which explains why grid losses decreased while MAE/WER worsened. We need to revert `main.py` to Minh’s loss (or switch both call sites to PyTorch CTC) before trusting new runs.
  2. **Decoder needs checkpoints.** To get meaningful WER we must load the same DS1 weights that produced the gradients. Add a `--decoder_checkpoint` flag to `src/analysis/generate_metrics.py` (or reuse `args.checkpoint_path`) so the decoder mirrors the teacher model.
  3. **Diagnostics before new runs.** Before launching another 4‑hour job, reuse the existing payload, rerun `generate_metrics.py` with the proper checkpoint, and confirm whether the decoded text still fails. If it does, consider a no-grid “full 10 s” run (set `--use_grid_optimization` off) purely as a sanity check.
  4. **Documentation updates.** Whenever we change the loss or decoder behavior, record the exact command + rationale in this README so we can track which experiments used Minh’s untouched objective versus the PyTorch variant.

## Experiment Log (append chronologically)
| Timestamp (EST)       | Action | Notes / Files |
|----------------------|--------|----------------|
| 2025-11-21 16:06     | Repo survey | Catalogued key scripts (`src/main.py`, `src/optimize.py`, datasets) to understand prior pipeline before new runs. |
| 2025-11-21 16:15     | Dataset prep | Generated `local_datasets/librispeech_long_10s/dataset_item_0.h5` via `python tmp_make_long_dataset.py` for the 10 s test case. |
| 2025-11-21 16:23     | Code change | Updated `src/main.py` with `_format_duration_label` so experiment folders auto-handle any min/max duration (required for 10 s). |
| 2025-11-21 16:35     | Runbook authoring | Created this README, captured baseline + long-form commands, and documented MAE/SNR/WER evaluation procedure. |
| 2025-11-21 16:37     | DS1 short-form run | `logging/0s-1s/DS1_batchstart_0_batch_end_1_...` finished with MAE≈2.6 and clean spectrograms/loss curves. |
| 2025-11-21 16:50     | DS1 long-form trial | First 10 s attempt (`logging/0s-13s/...`) hit NaNs during grid updates; added CTCLoss/`nan_to_num` guards for follow-up runs. |
| 2025-11-21 22:45     | Code change | Stabilized `batched_ctc_v2` with per-step normalization so DS1 gradients stay finite on the 10 s clip; verified gradients numerically. |
| 2025-11-21 22:47     | DS1 long-form sanity test | Ran 40-iteration grid sweep at `lr=0.05` (`logging/0s-13s/...lr_0.05...`) to confirm no NaNs before the full job. |
| 2025-11-21 22:55     | DS1 10 s full run (in progress) | Kicked off the 4000-iteration grid job with `grid_size=300`, `grid_overlap=150`; run now uses lisplab-1 RTX 6000 Ada GPU and is resumable via new checkpoints. |
| 2025-11-22 00:15     | Code change | Added per-grid figures + checkpoints (`src/optimize.py`) and documented the workflow; confirmed environment on lisplab-1 (`nvidia-smi`). |
| 2025-11-22 00:17     | DS1 long-form instrumentation test | Re-ran the 40-iteration job to validate checkpoint/plot filenames; all four grids saved PNGs + `sampleidx_0_grid_grid*_checkpoint.pt`. |
| 2025-11-22 00:18     | DS1 10 s full run (relaunch) | Started `timeout 14400s python ... --max_iterations 4000` via `nohup` (logs: `long_run.out`, pid in `long_run.pid`) so the job can finish unattended on lisplab-1 GPU 0. |
| 2025-11-22 01:40     | DS1 10 s evaluation | Run finished (runtime ≈4,767 s). Logged global MAE=8.01, SNR≈‑0.04 dB plus per-grid metrics + MFCC comparison plots under `analysis/`. Waveform reconstruction via `librosa` is blocked by the cluster’s `coverage` shim (numba import error), so decoder/WER are pending once that dependency issue is resolved. |
| 2025-11-22 02:00     | Code change | Upgraded `coverage` to 7.6.1 to unblock `librosa`/`numba`, added per-grid optimizer/scheduler resets + scheduler guards in `src/optimize.py`, and renamed the prior long-form folder to `..._preLRdecay` before the new run. |
| 2025-11-22 02:05     | DS1 10 s rerun | Re-launched `timeout 4h python src/main.py ... --use_grid_optimization` with the new LR reset logic; checkpoints + logs live in `logging/0s-13s/DS1_batchstart_0_batch_end_1_init_uniform_opt_Adam_lr_0.05_reg_None_regw_0.0_top-grad-perc_1.0_cpt_None/long_run.out` (runtime ≈4,656 s). |
| 2025-11-22 03:30     | Metrics/spectrograms | Added `src/analysis/generate_metrics.py` to automate MAE/SNR/WER/audio dumps. Generated fresh PNGs, `segment_metrics.csv`, `metrics.json`, and `sampleidx_0_reconstruction.wav` under the new experiment’s `analysis/` folder (global MAE ≈10.73, SNR ≈‑1.80 dB, decoder WER still 1.0 with random DS1 weights). |
| 2025-11-22 04:10     | Post-mortem | Confirmed the PyTorch-vs-Minhs CTC mismatch plus the “decoder uses random weights” bug, and documented both in the README along with next steps (load checkpoint during decoding, rerun with Minh’s loss, and optionally test a no-grid 10 s optimization). |

Add a new row every time you tweak hyper-parameters, rerun `main.py`, or touch
source files so we can reconstruct the timeline later.

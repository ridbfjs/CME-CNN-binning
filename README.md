# CME-CNN-binning

A PyTorch experiment for predicting the remaining Sun–Earth transit time of a coronal mass ejection (CME) from a LASCO coronagraph image and 11 physical features. The target is the interval, in hours, from the image timestamp to the recorded arrival time. The training script compares hard and Gaussian-soft bin targets for the same Model B architecture.

## Models and experiment

Model A maps a single-channel 256 × 256 image to a scalar using three convolutional blocks. Model B concatenates that scalar with 11 features covering CME kinematics, geometry, and six-hour OMNI measurements. Its fusion network has hidden layers of 16 and 32 units. The standalone Model B returns a scalar. The experiment replaces its output layer with a histogram head over 0.5-hour bins, with the upper bound determined from training targets.

The hard-target run assigns each observation to one bin. The soft-target run integrates a Gaussian centered on the observation (σ = 0.5 h) over the bin edges and normalizes the resulting masses over the modeled range. Both runs use the same fresh initialization, train the entire network, and use fold 1. Each method runs one smoke-test epoch and then a separately initialized 500-epoch formal run. The runner selects checkpoints by validation MAE of the distribution mean and reports results with the original and train-only recalibrated batch normalization statistics. The held-out test set is reserved and is not evaluated by this runner.

The paired binning runs provide a baseline for future knowledge distillation into a smaller model. This release contains the full Model B training workflow.

## Repository layout

| File | Purpose |
| --- | --- |
| [data_process.py](data_process.py) | Validates event and image metadata, produces review panels, masks and resizes images, and writes labels. |
| [quality_filter.py](quality_filter.py) | Applies reviewed exclusions and checks the quality-policy hash against preprocessing metadata. |
| [dataset.py](dataset.py) | Loads grayscale CNN inputs and transit-time labels. |
| [dataset_b.py](dataset_b.py) | Joins physical features by event ID and fits the scaler using training events. |
| [model_loader.py](model_loader.py) | Defines the image-only Model A. |
| [model_b.py](model_b.py) | Defines the image and physical-feature fusion model. |
| [run_full_modelb.py](run_full_modelb.py) | Runs the paired hard- and soft-binning experiments and records metrics and checkpoints. |

## Data preparation

Use Python 3.9 or later with NumPy, Pillow, openpyxl, and PyTorch. The full experiment requires a CUDA-capable PyTorch installation and an NVIDIA GPU. Use a PyTorch build compatible with your Python and CUDA environment.

The preprocessing script expects event directories named `event_<id>_<YYYYMMDD>_<HHMMSS>`, each containing PNG images and an `images.csv` manifest. The supplied workbook must contain the `Final_trainable_148` sheet with `event_id`, `CME_time_UT`, and `arrival_time_UT` columns. Place the original reviewed `quality_exclusions.json` next to the scripts. Its `groups` entries contain `event_folder`, `filenames`, and `reason`.

Run sample mode first and inspect the before/after comparison panels. After review, run batch mode with a fresh output directory outside the original image tree.

```bash
python data_process.py --root /path/to/LASCO_148 --excel /path/to/events.xlsx \
  --output /path/to/review_samples --mode sample

python data_process.py --root /path/to/LASCO_148 --excel /path/to/events.xlsx \
  --output /path/to/dataset --mode batch --review-approved
```

Batch mode writes processed images, `image_labels.csv`, `preprocessing.json`, and CSV reports for negative remaining times and reviewed image exclusions.

## Running the experiment

The repository contains the source code. Reproducing the experiment requires the LASCO imagery, event workbook, original quality policy, and experiment artifacts listed below, which are supplied separately.

| Runner input | Required files |
| --- | --- |
| Preprocessed data directory (`D`) | `image_labels.csv`, `preprocessing.json`, and the processed images referenced by the manifest |
| Fold-artifact directory (`T`) | `physical_148_model_b.csv`, `splits.json`, `scaler_01.json`, `fold_01.pt`, and `config.json` |
| Run directory (`R`) | A `code/` subdirectory with these scripts and `quality_exclusions.json` |

The runner uses paths from the original deployment. Place the scripts under `<run-directory>/code/` and configure `W`, `S`, `D`, and `T` at the top of the runner for your environment. The run directory must be under `W`, and resolved inputs must reside under `S`, `W`, or `D`. The runner verifies the saved scaler against the fold's training events and hashes the required artifacts, including the prior fold checkpoint. The new models start from the shared fresh initialization.

From the configured run directory, execute:

```bash
python code/run_full_modelb.py
```

Read `protocol.json` for the recorded settings, `status.json` for progress, and `results.csv` for the formal validation summary. The `smoke/` and `formal/` directories hold per-method training histories, checkpoints, validation predictions, metrics, and timing and memory measurements. A rerun in the same directory can resume from the saved `last.pt` for each phase and method.

# Notes

## Script fixes

- `bash/test_clear_human.sh`
  - Added an explicit CUDA preflight failure exit.
  - Moved `mname` computation to after `getopts` so `-m` overrides produce correct W&B names.
  - Removed `%a` from Slurm log filenames for non-array runs.
  - Updated W&B tags from `human,aug,clear` to `human,test,eval,clear`.

- `bash/train_clear_2_human.sh`
  - Removed `%a` from Slurm log filenames for non-array runs.

## ProstT5 / clear_2 issue

- The `KeyError: 'P24592'` during `train_clear_2.py` was caused by the ProstT5 loader using the wrong path layout.
- The code was looking for flat files like:
  - `/work/wl324/prostt5/human/P24592.pt`
- The actual files are stored in bucketed directories like:
  - `/work/wl324/prostt5/human/P/P2/P24592.pt`
- Fixed in:
  - `dscript/commands/train_clear_2.py`
  - `dscript/commands/evaluate_clear_2.py`
- Both now use the bucketed path helper and fail early with a clear missing-embedding message.
- `clear_2` now supports selecting either ESM2 or ProstT5 as the embedding source.
- `clear_2` does not concatenate ESM2 and ProstT5 together.
- Behavior:
  - default: use ESM2 embeddings
  - with `--allow_prostt5`: use ProstT5 embeddings instead of ESM2

## Model / training notes

- `clear`
  - Supports Lookahead only when `--use-lookahead` is passed.
  - Current `bash/train_clear_human.sh` does not pass `--use-lookahead`, so it trains with plain AdamW, the model is '/hpc/home/wl324/projects/tt3d/data/results/human_esm2_train_clear_lr0.0005_wd0.0001_dp0.2_grid128_d128/'

- `one_op` vs `opt`
  - In the current repo, `interaction_one_op.py` and `interaction_opt.py` are effectively identical.
  - The practical differences are in training setup:
    - `opt` uses Lookahead on top of AdamW.
    - `opt` exposes `--classifier-d-model`.
    - `one_op` uses plain AdamW.
  - In `bash/train_one_op_human.sh`, the shell variable `d=64` is only used in naming and is not passed to the Python command.

## Paths referenced

- ESM2 embeddings:
  - `/work/wl324/esm2/human`
- ProstT5 embeddings:
  - `/work/wl324/prostt5/human`
- Human FASTA:
  - `/hpc/home/wl324/projects/tt3d/data_archive/fasta/dscript_new/human.fasta`

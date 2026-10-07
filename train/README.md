# Fine-tuning the dust detector on your own scans

The app finds dust with a U-Net (`src/image_processing.py`) trained on synthetic dust
(`main.ipynb`). If you have scans together with their manually cleaned versions, you can
fine-tune it on real dust from your scanner and film.

## 1. Prepare pairs

Each pair is the **same scan** before and after cleaning, same size, not cropped, rotated or
color-corrected after cleaning. Only the dust spots may differ.

```
train_data/
  dusty/   frame01.tif   frame02.jpg ...
  clean/   frame01.tif   frame02.jpg ...
```

Files are matched by name (the extension may differ). 20–30 pairs is enough to start.
Images are ignored by git, so `train_data/` won't be committed.

## 2. Check the masks

```
pip install torch opencv-python numpy pillow
python train/finetune.py --data train_data --masks-only
```

The dust mask is `|dusty - clean| > --diff-threshold` (0.04 by default, about 10 levels of 255).
Open `train_data/_masks/*_mask.jpg`: dust is red.

- Red covering large areas or edges of objects → the pair is misaligned or the tones changed.
  Fix or remove that pair.
- Faint dust not marked → lower `--diff-threshold` (e.g. 0.025).
- Grain/JPEG noise marked → raise it (e.g. 0.06).

## 3. Train

Put the stock weights in `src/weights/` first (see `WINDOWS_BUILD.md`), then:

```
python train/finetune.py --data train_data
```

- A few pairs are held out for validation. The score before training is printed first, then
  per epoch; the best epoch is saved to `src/weights/finetuned_unet.pth`.
  If no epoch beats the stock model, nothing is saved.
- An NVIDIA GPU (CUDA) or Apple Silicon (MPS) is strongly recommended. On CPU it works but
  takes hours; try `--size 512 --epochs 8`.
- Out of GPU memory → `--size 512`.
- 30% of the samples get synthetic dust on your cleaned scans so the model doesn't forget what
  it already knows (`--synthetic 0` disables it).

All options: `python train/finetune.py --help`.

## 4. Use it

The app loads `src/weights/finetuned_unet.pth` in preference to the stock weights and the
PyInstaller builds bundle it along with the other `.pth` files. To go back to the stock model,
delete or rename the file.

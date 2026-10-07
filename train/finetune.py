#!/usr/bin/env python3
"""
Fine-tune the Spotless-Film dust detector (U-Net) on your own scans.

Input: pairs of the SAME scan before and after manual cleaning, pixel-aligned.

    train_data/
        dusty/  frame01.tif  frame02.jpg ...
        clean/  frame01.tif  frame02.jpg ...   (matched by file name, extension may differ)

The dust mask for each pair is |dusty - clean| > --diff-threshold. Previews of the
masks are written to <data>/_masks so you can check them before trusting the result.

Training starts from the existing weights, holds out a few pairs for validation and
saves the best epoch to src/weights/finetuned_unet.pth, which the app prefers over
the stock v5 weights. Delete that file to go back to the stock model.

    python train/finetune.py --data train_data
    python train/finetune.py --data train_data --masks-only   # just inspect the masks
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from image_processing import UNet  # noqa: E402  (same architecture the app loads)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
INFER_SIZE = 1024  # the app squeezes the whole image to 1024x1024 before inference


# --------------------------------------------------------------------------- data

def read_gray(path: Path) -> np.ndarray:
    """Read any 8/16-bit image as uint8 grayscale (cv2.imread fails on non-ASCII paths on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    return img.astype(np.uint8)


def find_pairs(data_dir: Path):
    dusty_dir, clean_dir = data_dir / "dusty", data_dir / "clean"
    if not dusty_dir.is_dir() or not clean_dir.is_dir():
        sys.exit(f"Expected {dusty_dir} and {clean_dir} folders")
    clean = {p.stem: p for p in clean_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs, missing = [], []
    for p in sorted(dusty_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem in clean:
            pairs.append((p, clean[p.stem]))
        else:
            missing.append(p.name)
    if missing:
        print(f"⚠️  No clean counterpart for: {', '.join(missing)}")
    return pairs


def build_mask(dusty: np.ndarray, clean: np.ndarray, threshold: float) -> np.ndarray:
    """Binary float mask where cleaning changed the image. Light blur suppresses JPEG noise."""
    d = cv2.GaussianBlur(dusty, (3, 3), 0).astype(np.float32)
    c = cv2.GaussianBlur(clean, (3, 3), 0).astype(np.float32)
    return (np.abs(d - c) / 255.0 > threshold).astype(np.float32)


def load_pairs(pairs, threshold: float, keep_side: int, preview_dir: Path):
    """Load every pair, build its mask, downscale to keep_side and save a preview."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for dusty_path, clean_path in pairs:
        try:
            dusty, clean = read_gray(dusty_path), read_gray(clean_path)
        except ValueError as e:
            print(f"⚠️  Skipped {dusty_path.name}: {e}")
            continue
        if dusty.shape != clean.shape:
            print(f"⚠️  Skipped {dusty_path.name}: sizes differ {dusty.shape[::-1]} vs {clean.shape[::-1]}")
            continue
        mask = build_mask(dusty, clean, threshold)
        coverage = mask.mean() * 100
        note = ""
        if coverage > 3:
            note = "  ← suspiciously large: pair misaligned or tones changed?"
        elif coverage == 0:
            note = "  ← empty: images identical or threshold too high"
        print(f"   {dusty_path.name}: {dusty.shape[1]}x{dusty.shape[0]}, dust {coverage:.3f}%{note}")

        overlay = cv2.cvtColor(dusty, cv2.COLOR_GRAY2BGR)
        overlay[mask > 0] = (0, 0, 255)
        cv2.imencode(".jpg", overlay)[1].tofile(str(preview_dir / f"{dusty_path.stem}_mask.jpg"))

        # Keep memory bounded: crops never need more than keep_side pixels per side.
        # INTER_AREA turns tiny specks into soft values, like the synthetic training masks.
        h, w = dusty.shape
        scale = min(1.0, keep_side / max(h, w))
        if scale < 1.0:
            size = (round(w * scale), round(h * scale))
            dusty = cv2.resize(dusty, size, interpolation=cv2.INTER_AREA)
            clean = cv2.resize(clean, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
        samples.append({"name": dusty_path.name, "dusty": dusty, "clean": clean, "mask": mask,
                        "dust_points": np.argwhere(mask > 0.25)})
    return samples


def generate_dust_mask(shape, num_blobs=250, num_scratches=4, num_hairs=50, max_blob_size=1,
                       max_scratch_length=240, max_hair_length=20, squiggliness=1.0, scale_factor=2):
    """Synthetic dust from main.ipynb (blobs, scratches, hairs), used to avoid forgetting."""
    h, w = shape
    H, W = h * scale_factor, w * scale_factor
    mask_hi = np.zeros((H, W), dtype=np.uint8)
    temp = np.zeros_like(mask_hi)

    def rand_count(base):
        return random.randint(int(base * 0.8), int(base * 1.2))

    def add(opacity):
        nonlocal mask_hi
        mask_hi = cv2.addWeighted(mask_hi, 1.0, temp, opacity, 0)

    max_blob_size = max(1, max_blob_size + random.randint(-1, 2))
    for _ in range(rand_count(num_blobs)):
        temp.fill(0)
        center = (random.randint(0, W), random.randint(0, H))
        axes = (random.randint(max(1, max_blob_size - 1), max_blob_size + 1) * scale_factor,
                random.randint(1, max(1, max_blob_size // 2) + 1) * scale_factor)
        cv2.ellipse(temp, center, axes, random.randint(0, 180), 0, 360, 255, -1)
        add(random.uniform(0.3, 1.0))

    for _ in range(rand_count(num_scratches)):
        temp.fill(0)
        x1, y1 = random.randint(0, W), random.randint(0, H)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(int(max_scratch_length * 0.7), int(max_scratch_length * 1.3)) * scale_factor
        x2, y2 = int(x1 + length * np.cos(angle)), int(y1 + length * np.sin(angle))
        cv2.line(temp, (x1, y1), (x2, y2), 255, scale_factor)
        add(random.uniform(0, 1.0))

    for _ in range(rand_count(num_hairs)):
        temp.fill(0)
        x, y = random.randint(0, W), random.randint(0, H)
        hair_length = random.randint(int(max_hair_length * 0.8), int(max_hair_length * 1.2)) * scale_factor
        segments = max(3, hair_length // random.randint(8, 12))
        sq = squiggliness * random.uniform(0.8, 1.2)
        angle = random.uniform(0, 2 * np.pi)
        points = []
        for _ in range(segments):
            seg = random.uniform(8, 12) * scale_factor
            x = int(np.clip(x + np.cos(angle) * seg + sq * random.uniform(-seg, seg), 0, W - 1))
            y = int(np.clip(y + np.sin(angle) * seg + sq * random.uniform(-seg, seg), 0, H - 1))
            points.append((x, y))
        cv2.polylines(temp, [np.array(points, dtype=np.int32)], False, 255,
                      random.randint(1, 2) * scale_factor, cv2.LINE_AA)
        add(random.uniform(0, 1.0))

    return cv2.resize(mask_hi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def random_crop(sample, size: int, crop_min: float, dust_focus: float):
    """Random crop resized to size x size. Crop fraction is relative to how much of the
    image the app sees in one 1024 inference, so dust keeps a realistic pixel scale.
    With probability dust_focus the crop is placed over a random dust pixel: dust covers
    ~0.1% of a scan, so plain random crops are mostly empty and teach "never any dust"."""
    img, clean, mask = sample["dusty"], sample["clean"], sample["mask"]
    h, w = img.shape
    frac = (size / INFER_SIZE) * random.uniform(crop_min, 1.0)
    ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
    points = sample["dust_points"]
    if len(points) and random.random() < dust_focus:
        py, px = points[random.randrange(len(points))]
        y = int(np.clip(py - random.randint(0, ch - 1), 0, h - ch))
        x = int(np.clip(px - random.randint(0, cw - 1), 0, w - cw))
    else:
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
    sl = (slice(y, y + ch), slice(x, x + cw))
    out = [cv2.resize(a[sl], (size, size), interpolation=cv2.INTER_AREA) for a in (img, clean, mask)]
    return out


def augment(img, mask):
    k = random.randint(0, 3)
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    if random.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    # mild exposure/contrast jitter, as between scans
    img = img.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * random.uniform(0.9, 1.1) + 0.5 + random.uniform(-0.05, 0.05), 0, 1)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def make_batch(samples, synth_pool, args):
    images, masks = [], []
    for _ in range(args.batch):
        sample = random.choice(samples)
        dusty, clean, mask = random_crop(sample, args.size, args.crop_min, args.dust_focus)
        if synth_pool and random.random() < args.synthetic:
            # synthetic dust on the cleaned version of your own scan
            synth = np.rot90(random.choice(synth_pool), random.randint(0, 3))
            synth = cv2.resize(np.ascontiguousarray(synth), (args.size, args.size))
            dusty = np.clip(clean.astype(np.float32) + synth * random.randint(150, 255), 0, 255).astype(np.uint8)
            mask = synth
        img, mask = augment(dusty, mask)
        images.append(img[None])
        masks.append(np.clip(mask, 0, 1)[None])  # resizing can overshoot 1.0 by float error
    return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


# ----------------------------------------------------------------------- training

def forward_logits(model, x):
    """Run the model but return the pre-sigmoid output of its last layer.

    UNet.forward applies sigmoid itself. Taking BCE of that probability kills the gradient
    exactly where the model is confidently wrong (missed dust, p ~ 1e-30), which is what
    fine-tuning has to fix, so the loss is computed from the logits instead.
    """
    captured = {}
    hook = model.final.register_forward_hook(lambda _m, _i, out: captured.update(z=out))
    try:
        model(x)
    finally:
        hook.remove()
    return captured["z"]


def loss_fn(logits, target, dice_weight):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if dice_weight == 0 or target.sum() < 1:
        return bce  # Dice on a dust-free batch would reward predicting nothing
    pred = torch.sigmoid(logits)
    # one Dice over the whole batch, so a single empty crop can't dominate it
    dice = 1 - 2 * (pred * target).sum() / (pred.sum() + target.sum())
    return bce + dice_weight * dice


@torch.no_grad()
def evaluate(model, samples, device, dice_weight):
    """Validate exactly like the app infers: whole image squeezed to 1024x1024."""
    model.eval()
    losses, tp, fp, fn = [], 0, 0, 0
    for s in samples:
        img = cv2.resize(s["dusty"], (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(s["mask"], (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
        y = torch.from_numpy(np.clip(mask, 0, 1))[None, None].to(device)
        logits = forward_logits(model, x)
        losses.append(loss_fn(logits, y, dice_weight).item())
        p, t = logits > 0, y > 0.2  # logit 0 == probability 0.5
        tp += (p & t).sum().item()
        fp += (p & ~t).sum().item()
        fn += (~p & t).sum().item()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return float(np.mean(losses)), f1


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "train_data", help="folder with dusty/ and clean/")
    ap.add_argument("--weights", type=Path, default=REPO / "src/weights/v5_bce_unet_epoch30.pth",
                    help="starting weights")
    ap.add_argument("--out", type=Path, default=REPO / "src/weights/finetuned_unet.pth")
    ap.add_argument("--diff-threshold", type=float, default=0.04,
                    help="min brightness change (0..1) counted as dust when building masks")
    ap.add_argument("--masks-only", action="store_true", help="build mask previews and exit")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=0, help="steps per epoch (default: 4 per training pair)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--size", type=int, default=1024, help="training crop size; use 512 if out of memory")
    ap.add_argument("--crop-min", type=float, default=0.6)
    ap.add_argument("--dust-focus", type=float, default=0.7, help="share of crops placed over dust")
    ap.add_argument("--val-count", type=int, default=-1, help="pairs held out (default: ~15%%, at least 2)")
    ap.add_argument("--synthetic", type=float, default=0.3, help="share of batches with synthetic dust")
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = find_pairs(args.data)
    if not pairs:
        sys.exit("No image pairs found")
    print(f"📂 {len(pairs)} pairs, building masks (threshold {args.diff_threshold})")
    samples = load_pairs(pairs, args.diff_threshold, keep_side=2 * INFER_SIZE,
                         preview_dir=args.data / "_masks")
    print(f"🖼  Mask previews (dust in red): {args.data / '_masks'}")
    if args.masks_only:
        return
    if len(samples) < 3:
        sys.exit("Need at least 3 usable pairs")

    random.shuffle(samples)
    val_count = args.val_count if args.val_count >= 0 else max(2, round(len(samples) * 0.15))
    val, train = samples[:val_count], samples[val_count:]
    print(f"🧪 train {len(train)}, validation {len(val)}: {', '.join(s['name'] for s in val)}")

    synth_pool = []
    if args.synthetic > 0:
        print("✨ Generating synthetic dust masks...")
        synth_pool = [generate_dust_mask((INFER_SIZE, INFER_SIZE), squiggliness=random.uniform(0.5, 1.5))
                      for _ in range(16)]

    device = pick_device(args.device)
    model = UNet()
    if not args.weights.exists():
        sys.exit(f"Starting weights not found: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device)
    print(f"🚀 Device: {device}, starting from {args.weights.name}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.steps or 4 * len(train)

    best_loss, best_f1 = evaluate(model, val, device, args.dice_weight)
    print(f"📏 Before fine-tuning: val loss {best_loss:.4f}, F1 {best_f1:.3f}")
    saved = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for step in range(steps):
            x, y = make_batch(train, synth_pool, args)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = forward_logits(model, x)
            loss = loss_fn(logits, y, args.dice_weight)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            print(f"\r   epoch {epoch}/{args.epochs} step {step + 1}/{steps} loss {total / (step + 1):.4f}",
                  end="", flush=True)

        val_loss, f1 = evaluate(model, val, device, args.dice_weight)
        mark = ""
        if val_loss < best_loss:
            best_loss, best_f1 = val_loss, f1
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            saved, mark = True, "  ✅ saved"
        print(f"\r   epoch {epoch}/{args.epochs}: train {total / steps:.4f}, val loss {val_loss:.4f}, "
              f"F1 {f1:.3f} ({time.time() - start:.0f}s){mark}")

    if saved:
        print(f"🎉 Best model: {args.out} (val loss {best_loss:.4f}, F1 {best_f1:.3f})")
        print("   The app will use it on next start. Delete the file to go back to the stock model.")
    else:
        print("🤷 No epoch beat the starting weights on validation; nothing saved.")


if __name__ == "__main__":
    main()

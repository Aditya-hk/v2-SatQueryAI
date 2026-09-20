"""BigEarthNet LoRA fine-tuning for the SatQuery vision-language encoder.

Remote-sensing adaptation (mandatory SIH-167 component): the compact ViT-style
encoder in :mod:`satquery.models.single_image_vqa` is fine-tuned with LoRA
adapters on BigEarthNet multi-label imagery. Sentinel-2 bands (B02 blue,
B03 green, B04 red, B08 NIR) are stacked as a 4-channel input; the same
recipe supports Sentinel-1+2 pairs through ``--channels 6`` with the fusion
encoder in :mod:`satquery.models.cross_modal_fusion`.

Usage
-----
    python -m satquery.training.fine_tune_bigearthnet \
        --bigearthnet-root /data/BigEarthNet-v2.0 \
        --epochs 5 --lora-rank 8 --batch-size 32 --out checkpoints

The dataset loader expects the standard BigEarthNet-19 layout: one folder per
patch (``S2A_T..._<id>``) containing ``*_B02.tif ... B08.tif`` plus a
``labels_19.json`` (or BigEarthNet-v2 ``*_labels_s2.json``). Metadata with
scene-level text annotations (BigEarthNet.txt-style CSV/JSONL) is used to
align the multi-label head with the caption grounding vocabulary. A small
synthetic fallback dataset is provided so the training loop can be verified
end-to-end without downloading the real corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from satquery.models.single_image_vqa import BIGEARTHNET_LABELS, LoRALinear, RSVisionEncoder
from satquery.utils.logger import get_logger

logger = get_logger("training")

NUM_LABELS = len(BIGEARTHNET_LABELS)


# ------------------------------------------------------------------ config ----
@dataclass
class TrainConfig:
    """Hyper-parameters for the LoRA adaptation run."""

    bigearthnet_root: Path = Path("data/BigEarthNet-v2.0")
    out_dir: Path = Path("satquery/models/checkpoints")
    epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    weight_decay: float = 0.01
    val_fraction: float = 0.1
    num_workers: int = 0
    seed: int = 42
    patch_size: int = 120          # BigEarthNet 120x120 patches
    max_samples: Optional[int] = None
    amp: bool = True
    log_every: int = 20
    channels: int = 4              # 4 = S2 B02/B03/B04/B08, 6 = S1+S2 pair mode
    synthetic_samples: int = 256   # size of the fallback synthetic dataset

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["bigearthnet_root"] = str(self.bigearthnet_root)
        payload["out_dir"] = str(self.out_dir)
        return payload


# ---------------------------------------------------------------- dataset -----
def _label_vector(raw_labels: Sequence[str]) -> np.ndarray:
    """Map BigEarthNet-19 label names to a multi-hot vector aligned with
    ``BIGEARTHNET_LABELS`` (19 classes in the canonical order used here)."""
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for label in raw_labels:
        label_norm = label.strip().lower()
        for idx, known in enumerate(BIGEARTHNET_LABELS):
            if known.lower() == label_norm:
                vec[idx] = 1.0
                break
    return vec


def _read_band(path: Path) -> Optional[np.ndarray]:
    """Read a single band as float32 in [0, 1] via rasterio or PIL/tifffile."""
    try:
        try:
            import rasterio

            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)
        except ImportError:
            pass
        try:
            import tifffile

            return tifffile.imread(path).astype(np.float32)
        except ImportError:
            pass
        from PIL import Image

        return np.asarray(Image.open(path), dtype=np.float32)
    except Exception as exc:
        logger.debug("Failed to read band %s: %s", path, exc)
        return None


def _read_patch_channels(patch_dir: Path, channels: int) -> Optional[np.ndarray]:
    """Load the 4 S2 bands (or 6-channel S1+S2 stack) for one patch."""
    if channels == 4:
        band_files = ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]
        band_paths: List[Path] = []
        for name in band_files:
            matches = sorted(patch_dir.glob(f"*{name}")) or sorted(patch_dir.glob(name))
            if not matches:
                return None
            band_paths.append(matches[0])
        arrays = [_read_band(p) for p in band_paths]
    else:  # 6-channel: S1 VV, VH + S2 B04, B03, B02, B08
        s1 = ["*_VV.tif", "*_VH.tif"]
        s2 = ["*_B04.tif", "*_B03.tif", "*_B02.tif", "*_B08.tif"]
        band_paths = []
        for pattern in s1 + s2:
            matches = sorted(patch_dir.glob(pattern))
            if not matches:
                return None
            band_paths.append(matches[0])
        arrays = [_read_band(p) for p in band_paths]
    if any(a is None for a in arrays):
        return None
    stacked = np.stack(arrays, axis=0).astype(np.float32)  # (C, H, W)
    # per-band 2nd-99th percentile stretch for stable inputs
    for c in range(stacked.shape[0]):
        band = stacked[c]
        lo, hi = np.percentile(band, 2), np.percentile(band, 99)
        stacked[c] = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return stacked


def _patch_label_file(patch_dir: Path) -> Optional[Path]:
    for candidate in ("labels_19.json", "labels.json"):
        if (patch_dir / candidate).exists():
            return patch_dir / candidate
    for pattern in ("*_labels_s2.json", "*_labels.json"):
        matches = sorted(patch_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_label_metadata(bigearthnet_root: Path) -> Dict[str, List[str]]:
    """Parse BigEarthNet.txt-style text annotations (CSV or JSONL) if present."""
    annotations: Dict[str, List[str]] = {}
    for name in ("BigEarthNet.txt", "metadata.csv", "metadata.jsonl", "labels.csv"):
        path = bigearthnet_root / name
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    patch_id = parts[0].strip()
                    labels = [p.strip() for p in parts[1:] if p.strip()]
                    annotations[patch_id] = labels
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    return annotations


class BigEarthNetDataset:
    """Folder-based BigEarthNet loader producing (image tensor, label vector)."""

    def __init__(self, root: Path, channels: int = 4, max_samples: Optional[int] = None) -> None:
        self.root = Path(root)
        self.channels = channels
        self.text_annotations = load_label_metadata(self.root)
        self.samples: List[Tuple[Path, List[str]]] = []
        for patch_dir in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not patch_dir.is_dir():
                continue
            label_file = _patch_label_file(patch_dir)
            labels: Optional[List[str]] = None
            if label_file is not None:
                try:
                    payload = json.loads(label_file.read_text(encoding="utf-8"))
                    labels = payload.get("labels", payload) if isinstance(payload, dict) else payload
                except Exception:
                    labels = None
            if labels is None:
                patch_id = patch_dir.name
                labels = self.text_annotations.get(patch_id)
            if labels:
                self.samples.append((patch_dir, list(labels)))
        if max_samples:
            self.samples = self.samples[:max_samples]
        logger.info("BigEarthNet loader indexed %d labelled patches under %s",
                    len(self.samples), self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for patch_dir, labels in self.samples:
            image = _read_patch_channels(patch_dir, self.channels)
            if image is None:
                continue
            yield image, _label_vector(labels)


class SyntheticBEDataset:
    """Deterministic synthetic stand-in with BigEarthNet-19 label semantics.

    Used when the real corpus is unavailable, so the LoRA training path can
    be executed and verified end-to-end (and serves as a CI fixture).
    """

    def __init__(self, n_samples: int = 256, size: int = 120, seed: int = 42,
                 channels: int = 4) -> None:
        rng = np.random.default_rng(seed)
        self.channels = channels
        self.items: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_samples):
            # 4 planes: B02 blue, B03 green, B04 red, B08 NIR (S2 convention)
            scene = rng.random((4, size, size)).astype(np.float32) * 0.4
            mode = rng.integers(0, 3)
            if mode == 0:      # vegetation-ish: bright NIR + green
                scene[1] += 0.4
                scene[3] += 0.35
                labels = ["Natural grassland", "Arable land"]
            elif mode == 1:    # urban-ish: bright red, flat spectrum
                scene[2] += 0.35
                scene[1] += 0.2
                labels = ["Urban fabric", "Industrial or commercial units"]
            else:              # water-ish: bright blue, dark NIR
                scene[0] += 0.4
                labels = ["Inland waters", "Coastal wetlands"]
            if channels <= 4:
                image = np.clip(scene[:channels], 0.0, 1.0).astype(np.float32)
            else:  # 6-channel S1+S2 mode: append VV/VH planes
                extra = rng.random((channels - 4, size, size)).astype(np.float32) * 0.3
                image = np.clip(np.concatenate([scene, extra]), 0.0, 1.0).astype(np.float32)
            self.items.append((image, _label_vector(labels)))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield from self.items


# -------------------------------------------------------------- LoRA engine ---
def inject_lora_adapters(encoder: RSVisionEncoder, rank: int = 8, alpha: float = 16.0) -> List[str]:
    """Attach LoRA adapters to every ``nn.Linear`` in the encoder core.

    Uses forward hooks (plus ``lora_A`` / ``lora_B`` parameters stored on the
    target module) instead of wrapping the modules: ``nn.TransformerEncoderLayer``
    fast paths reach into ``linear.weight`` directly, so wrapper modules would
    break them. At inference time the trained adapters are merged into the base
    weights by :meth:`RSVisionEncoder.load_bigearthnet_adapter`, which is
    mathematically identical to the hook path.
    """
    import torch
    import torch.nn as nn

    try:  # keep the eval fast path from silently bypassing the hooks
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass

    injected: List[str] = []
    targets: List[Tuple[str, nn.Linear]] = []
    for block_idx, block in enumerate(encoder.core.blocks.layers):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                targets.append((f"block{block_idx}.{name}", module))
    for path, module in targets:
        dev = module.weight.device  # match the device the encoder was moved to
        module.lora_A = nn.Parameter(torch.randn(rank, module.in_features, device=dev) * 0.02)
        module.lora_B = nn.Parameter(torch.zeros(module.out_features, rank, device=dev))
        scaling = alpha / max(1, rank)

        def make_hook(mod: nn.Linear, scale: float):
            def hook(_module, inputs, output):
                x = inputs[0]
                return output + (x @ mod.lora_A.T @ mod.lora_B.T) * scale
            return hook

        module.register_forward_hook(make_hook(module, scaling))
        injected.append(path)
    return injected


def lora_parameters(encoder: RSVisionEncoder) -> List[Any]:
    """Only LoRA A/B matrices + the label head are trainable."""
    trainable: List[Any] = []
    for module in encoder.core.modules():
        for attr in ("lora_A", "lora_B"):
            param = getattr(module, attr, None)
            if isinstance(param, torch_Parameter()):
                trainable.append(param)
    trainable += list(encoder.multi_label_head.parameters())
    return trainable


def torch_Parameter():
    import torch.nn as nn

    return nn.Parameter


def freeze_base_weights(encoder: RSVisionEncoder) -> None:
    """Freeze the pretrained trunk; LoRA adapters + label head stay trainable."""
    import torch.nn as nn

    for param in encoder.core.parameters():
        param.requires_grad_(False)
    for module in encoder.core.modules():
        for attr in ("lora_A", "lora_B"):
            param = getattr(module, attr, None)
            if isinstance(param, nn.Parameter):
                param.requires_grad_(True)
    for param in encoder.multi_label_head.parameters():
        param.requires_grad_(True)


def split_train_val(dataset: Any, val_fraction: float,
                    seed: int) -> Tuple[List[Any], List[Any]]:
    items = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_fraction)) if items else 0
    return items[n_val:], items[:n_val]


def batches(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _accuracy_at_k(probs: np.ndarray, targets: np.ndarray, k: int = 3) -> float:
    if targets.shape[0] == 0:
        return 0.0
    hits = 0
    for p, t in zip(probs, targets):
        top = np.argsort(p)[::-1][:k]
        hits += 1 if t[top].sum() > 0 else 0
    return hits / targets.shape[0]


def train_lora(config: TrainConfig, device: Optional[str] = None) -> Dict[str, Any]:
    """Run LoRA fine-tuning of the RS encoder; saves the merged state dict.

    The saved checkpoint (``satquery_encoder_ben_lora.pt``) is picked up
    automatically by ``RSVisionEncoder`` at inference time, completing the
    remote-sensing adaptation loop.
    """
    import torch
    import torch.nn.functional as F

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                     ("mps" if torch.backends.mps.is_available() else "cpu")))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset: Any
    if config.bigearthnet_root.exists() and any(config.bigearthnet_root.iterdir()):
        dataset = BigEarthNetDataset(config.bigearthnet_root, channels=config.channels,
                                     max_samples=config.max_samples)
        if len(dataset) == 0:
            logger.warning("No labelled patches found under %s; falling back to synthetic data.",
                           config.bigearthnet_root)
            dataset = SyntheticBEDataset(channels=config.channels, seed=config.seed,
                                     n_samples=config.synthetic_samples)
    else:
        logger.info("BigEarthNet root %s not present; using deterministic synthetic "
                    "BigEarthNet-style data for the adaptation run.", config.bigearthnet_root)
        dataset = SyntheticBEDataset(channels=config.channels, seed=config.seed,
                                     n_samples=config.synthetic_samples)

    train_items, val_items = split_train_val(dataset, config.val_fraction, config.seed)
    encoder = RSVisionEncoder(in_channels=config.channels, enable_lora_adapter=False).to(device)
    injected = inject_lora_adapters(encoder, rank=config.lora_rank, alpha=config.lora_alpha)
    freeze_base_weights(encoder)
    trainable = lora_parameters(encoder)
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(enabled=config.amp and device.type == "cuda")

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    global_step = 0
    encoder.train()
    for epoch in range(1, config.epochs + 1):
        running_loss = 0.0
        steps = 0
        for batch in batches(train_items, config.batch_size):
            images = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
            targets = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                tokens = encoder.core(images)
                pooled = tokens.mean(dim=1)
                logits = encoder.multi_label_head(pooled)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            steps += 1
            global_step += 1
            if steps % config.log_every == 0:
                logger.info("epoch %d step %d loss %.4f", epoch, global_step,
                            running_loss / steps)

        # ---------------- validation ----------------
        encoder.eval()
        val_loss, val_hits, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in batches(val_items, config.batch_size):
                images = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
                targets = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
                tokens = encoder.core(images)
                pooled = tokens.mean(dim=1)
                logits = encoder.multi_label_head(pooled)
                val_loss += float(F.binary_cross_entropy_with_logits(logits, targets).item())
                probs = torch.sigmoid(logits).cpu().numpy()
                val_hits += _accuracy_at_k(probs, targets.cpu().numpy(), k=3)
                n_val += 1
        encoder.train()
        val_loss = val_loss / max(1, n_val)
        val_acc = val_hits / max(1, n_val)
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, steps),
                        "val_loss": val_loss, "val_acc@3": round(val_acc, 4)})
        logger.info("epoch %d done: train %.4f val %.4f acc@3 %.3f",
                    epoch, running_loss / max(1, steps), val_loss, val_acc)
        if val_loss < best_val:
            best_val = val_loss
            config.out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = config.out_dir / "satquery_encoder_ben_lora.pt"
            torch.save(encoder.state_dict(), ckpt)

    # final checkpoint + metadata for the audit trail
    config.out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = config.out_dir / "satquery_encoder_ben_lora.pt"
    torch.save(encoder.state_dict(), final_ckpt)
    metadata = {
        "encoder": "satquery RS ViT-B/16 (compact)",
        "method": "LoRA (rank {}, alpha {})".format(config.lora_rank, config.lora_alpha),
        "injected_modules": injected,
        "dataset": type(dataset).__name__,
        "labels": list(BIGEARTHNET_LABELS),
        "history": history,
        "config": config.to_dict(),
    }
    (config.out_dir / "satquery_encoder_ben_lora.meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved LoRA checkpoint to %s", final_ckpt)
    return {
        "checkpoint": str(final_ckpt),
        "best_val_loss": best_val,
        "history": history,
        "injected_modules": injected,
        "n_train": len(train_items),
        "n_val": len(val_items),
    }


# ------------------------------------------------------------------- CLI ------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BigEarthNet LoRA fine-tuning for SatQuery AI")
    parser.add_argument("--bigearthnet-root", type=Path, default=TrainConfig.bigearthnet_root)
    parser.add_argument("--out", type=Path, default=TrainConfig.out_dir)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--channels", type=int, default=4, choices=(4, 6))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = TrainConfig(
        bigearthnet_root=args.bigearthnet_root,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_samples=args.max_samples,
        channels=args.channels,
        amp=not args.no_amp,
        seed=args.seed,
    )
    report = train_lora(config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the single-seed metrics and plots used by the report entry points."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pet_experiment import HERE, MODELS, execute, scores


def run(size, gallery=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Replay existing checkpoints")
    parser.add_argument("--output", type=Path, default=HERE / "course_check" if gallery else HERE)
    parser.add_argument("--runs", type=Path, default=HERE / "pet_runs")
    args = parser.parse_args()
    models = ("UNet",) if gallery else MODELS
    records, preds, data, splits, images, masks = execute(
        size, 2026, args.runs / f"size{size}" / "seed2026", models, args.verify)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    prefix = "s96_" if size == 96 else ""
    (out / (prefix + "segmentation.json")).write_text(json.dumps(records, indent=2))
    (out / (prefix + "seg_split.json")).write_text(json.dumps(splits))
    ids = splits[2]
    winner = max(records, key=lambda name: max(records[name]["curve"]))
    pred = preds[winner]
    if gallery:
        np.savez_compressed(out / "examples.npz", rgb=np.stack([images[k] for k in ids]),
                            truth=np.stack([masks[k] for k in ids]), pred=pred.cpu().numpy(), ids=ids)
        return
    truth = data[2][1]
    metrics = [scores(p, t) for p, t in zip(pred, truth)]
    order = sorted(range(len(ids)), key=lambda i: metrics[i]["miou"])
    examples = []
    fig, rows = plt.subplots(3, 4, figsize=(9, 7))
    for row, i in zip(rows, [order[0], order[len(ids)//2], order[-1]]):
        key = ids[i]
        examples.append([key, metrics[i]])
        panels = [images[key], np.where(masks[key] == 3, .5, masks[key] == 1),
                  pred[i].cpu(), ((pred[i] != truth[i]) & (truth[i] != 255)).cpu()]
        for ax, panel, title in zip(row, panels, ["Input", "Ground truth", "Prediction", "Error"]):
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1 if title != "Input" else 255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / (prefix + "seg_examples.png"), dpi=160)
    plt.close(fig)
    (out / (prefix + "seg_examples.json")).write_text(json.dumps(dict(model=winner, examples=examples)))
    fig, ax = plt.subplots()
    for name, record in records.items():
        ax.plot(range(1, len(record["curve"])+1), record["curve"], label=name)
    ax.set(xlabel="Epoch", ylabel="Validation mIoU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / (prefix + "seg_curve.png"), dpi=165)
    plt.close(fig)

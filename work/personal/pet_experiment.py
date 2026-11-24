"""Shared Pet training, checkpoint replay and paired multi-seed evaluation."""
from pathlib import Path
import argparse
import io
import json
import os
import tarfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ("EncoderDecoder", "UNet", "UNetDice")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_data(size):
    splits = json.loads((ROOT / "reference/seg_split.json").read_text())
    names = sum(splits, [])
    folder = ROOT / "work/provenance/downloads"
    images = {}
    with tarfile.open(folder / "images.tar.gz.incomplete", "r|gz") as archive:
        for member in archive:
            key = Path(member.name).stem
            if member.name.endswith(".jpg") and key in names:
                image = Image.open(io.BytesIO(archive.extractfile(member).read())).convert("RGB")
                images[key] = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))
            if len(images) == len(names):
                break
    masks = {}
    with tarfile.open(folder / "annotations-full.tar.gz") as archive:
        for key in names:
            raw = archive.extractfile(f"annotations/trimaps/{key}.png").read()
            masks[key] = np.asarray(Image.open(io.BytesIO(raw)).resize((size, size), Image.Resampling.NEAREST))
    data = []
    for split in splits:
        x = torch.as_tensor(np.stack([images[k] for k in split]),device="cuda").permute(0,3,1,2).float()/255
        y = torch.as_tensor(np.stack([masks[k] for k in split]),device="cuda")
        data.append((x, torch.where(y == 3, 255, (y == 1).long())))
    return data, splits, images, masks


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU())


class PetNet(nn.Module):
    def __init__(self, skip):
        super().__init__()
        self.skip = skip
        self.a, self.b, self.c = block(3,16), block(16,32), block(32,64)
        self.d, self.e = block(64 + 32*skip,32), block(32 + 16*skip,16)
        self.f = nn.Conv2d(16,2,1)

    def up(self, x, side):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.cat([x, side], 1) if self.skip else x

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.max_pool2d(a,2))
        c = self.c(F.max_pool2d(b,2))
        return self.f(self.e(self.up(self.d(self.up(c,b)),a)))


def scores(pred, target):
    valid = target != 255
    cm = torch.bincount(2*target[valid]+pred[valid], minlength=4).reshape(2,2).cpu().numpy()
    correct = cm.diagonal()
    iou = correct / (cm.sum(0)+cm.sum(1)-correct)
    return dict(cm=cm.tolist(), iou=iou.tolist(), miou=float(iou.mean()),
                accuracy=float(correct.sum()/cm.sum()), correct=int(correct.sum()), pixels=int(cm.sum()))


@torch.no_grad()
def predict(net, images):
    net.eval()
    return torch.cat([net(x).argmax(1) for x in images.split(16)])


def train(data, name, seed, directory):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = PetNet(name != "EncoderDecoder").cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    images, targets = data[0]
    curve, best = [], -1
    for epoch in range(1,26):
        model.train()
        for indices in torch.randperm(len(images)).split(8):
            target = targets[indices]
            output = model(images[indices])
            loss = F.cross_entropy(output.permute(0, 2, 3, 1).reshape(-1, 2), target.flatten(), ignore_index=255)
            if name == "UNetDice":
                a = output.softmax(1)[:,1] * (target != 255)
                b = target == 1
                loss = loss + 1 - (2*(a*b).sum()+1)/(a.sum()+b.sum()+1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation = scores(predict(model, data[1][0]), data[1][1])["miou"]
        curve.append(validation)
        if validation > best:
            best = validation
            torch.save(dict(model=name, epoch=epoch, seed=seed, size=images.shape[-1],
                            state=model.state_dict()), directory / "best.pt")
    pred, ckpt = replay(directory, data[2][0])
    metrics = dict(**scores(pred,data[2][1]), epoch=ckpt["epoch"], curve=curve,
                   params=sum(p.numel() for p in model.parameters()))
    np.savez_compressed(directory/"prediction.npz", pred=pred.cpu().numpy().astype(np.uint8))
    (directory/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return pred, metrics


def replay(directory, images):
    ckpt = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
    model = PetNet(ckpt["model"] != "EncoderDecoder").cuda()
    model.load_state_dict(ckpt["state"])
    return predict(model,images), ckpt


def execute(size, seed, directory, models=MODELS, verify=False):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data, splits, images, masks = load_data(size)
    results, predictions = {}, {}
    for name in models:
        folder = directory / name
        if verify:
            pred, ckpt = replay(folder, data[2][0])
            record = json.loads((folder/"metrics.json").read_text())
            assert (ckpt["model"],ckpt["size"],ckpt["seed"]) == (name,size,seed)
            assert np.array_equal(pred.cpu().numpy(), np.load(folder/"prediction.npz")["pred"])
            assert all(record[k] == v for k,v in scores(pred,data[2][1]).items())
            assert record["epoch"] == ckpt["epoch"] == 1+int(np.argmax(record["curve"]))
        else:
            pred, record = train(data,name,seed,folder)
        print(size,seed,name,record["epoch"],record["miou"],flush=True)
        results[name], predictions[name] = record,pred
    return results,predictions,data,splits,images,masks


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--output",type=Path,default=HERE/"pet_runs")
    args=parser.parse_args()
    for size in (128,96):
        for seed in (2024,2025,2026):
            execute(size,seed,args.output/f"size{size}"/f"seed{seed}",verify=args.verify)


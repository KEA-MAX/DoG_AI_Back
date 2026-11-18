"""DogFLW sample format -> YOLO Pose; CPU training and inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil


def annotation(image, label):
    from PIL import Image
    with Image.open(image) as im:
        w, h = im.size
    d = json.loads(Path(label).read_text(encoding="utf-8-sig"))
    pts = d["landmarks"]
    if len(pts) != 46 or any(len(p) != 2 for p in pts):
        raise ValueError(f"Expected 46 XY landmarks: {label}")
    pts = [(float(x), float(y)) for x, y in pts]
    x1, y1, x2, y2 = map(float, d["bounding_boxes"])
    if not all(math.isfinite(v) for p in pts for v in p):
        raise ValueError(f"Non-finite landmark: {label}")
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Invalid/outside bounding box: {label}")
    if any(not (0 <= x <= w and 0 <= y <= h) for x, y in pts):
        raise ValueError(f"Outside landmark (not silently clipped): {label}")
    values = [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]
    values += [v for x, y in pts for v in (x/w, y/h)]
    return "0 " + " ".join(f"{v:.8f}" for v in values) + "\n", pts, (x1,y1,x2,y2)


def preview(a):
    from PIL import Image, ImageDraw
    _, pts, box = annotation(a.image, a.json)
    with Image.open(a.image) as src:
        im = src.convert("RGB")
    x1,y1,x2,y2 = box
    crop = (max(0,int(x1)-10),max(0,int(y1)-10),min(im.width,int(x2)+11),min(im.height,int(y2)+11))
    im = im.crop(crop).resize(((crop[2]-crop[0])*6,(crop[3]-crop[1])*6))
    draw = ImageDraw.Draw(im)
    for i,(x,y) in enumerate(pts):
        x,y = (x-crop[0])*6,(y-crop[1])*6
        draw.ellipse((x-3,y-3,x+3,y+3), fill="lime")
        draw.text((x+4,y-12),str(i),fill="yellow",stroke_width=1,stroke_fill="black")
    im.save(a.out)
    print(f"46 points; saved {a.out}")


def collect(root):
    images = root / "images"
    labels = root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise ValueError(f"Required: {images} and {labels}")
    result = []
    seen = set()
    for im in sorted(images.iterdir()):
        if im.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        key = im.stem.lower()
        if key in seen:
            raise ValueError(f"Duplicate image stem: {im}")
        seen.add(key)
        label = labels / (im.stem + ".json")
        row, _, _ = annotation(im, label)
        result.append((im, row))
    if not result:
        raise ValueError(f"No images: {images}")
    return result


def convert(a):
    root, out = Path(a.source).resolve(), Path(a.out).resolve()
    if out.exists():
        raise ValueError(f"Output already exists; choose a new --out: {out}")
    train = collect(root / "train")
    if len(train) < 2:
        raise ValueError("At least two training images required for a split")
    random.Random(a.seed).shuffle(train)
    n = max(1, min(len(train)-1, round(len(train)*a.val_fraction)))
    splits = {"train":train[n:], "val":train[:n]}
    if (root / "test").exists():
        splits["test"] = collect(root / "test")
    # Reject exact duplicate bytes anywhere, before writing. Individual identity
    # and near-duplicate detection require separate dataset review.
    hashes = {}
    for split, records in splits.items():
        for im, _ in records:
            digest = hashlib.sha256(im.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image bytes: {im} and {hashes[digest]}")
            hashes[digest] = str(im)
    manifest = {}
    for split, records in splits.items():
        for folder in ("images", "labels"):
            (out/folder/split).mkdir(parents=True, exist_ok=True)
        manifest[split] = []
        for im, row in records:
            shutil.copy2(im, out/"images"/split/im.name)
            (out/"labels"/split/(im.stem+".txt")).write_text(row,encoding="utf-8")
            manifest[split].append(str(im))
    yaml = "path: " + json.dumps(out.as_posix(),ensure_ascii=False) + "\n"
    yaml += "train: images/train\nval: images/val\n"
    if "test" in splits:
        yaml += "test: images/test\n"
    yaml += "kpt_shape: [46, 2]\nnames:\n  0: dog_face\n"
    (out/"data.yaml").write_text(yaml,encoding="utf-8")
    (out/"split_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print({k:len(v) for k,v in splits.items()})
    print(out/"data.yaml")


def train(a):
    from ultralytics import YOLO
    YOLO(a.model).train(data=str(Path(a.data).resolve()), device="cpu",
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=0,
        fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0,
        project="runs/dogface", name="train", seed=42)


def predict(a):
    from ultralytics import YOLO
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    results = YOLO(a.weights).predict(source=a.source, device="cpu",
        imgsz=a.imgsz, conf=a.conf, stream=True)
    for i,r in enumerate(results):
        name = f"{i:06d}_{Path(r.path).stem}"
        r.save(filename=str(out/(name+".jpg")), kpt_line=False)
        detections = []
        if r.keypoints is not None:
            for box, conf, points in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist(),r.keypoints.xy.tolist()):
                detections.append({"box_xyxy":box,"detection_confidence":conf,"landmarks_xy":points})
        (out/(name+".json")).write_text(json.dumps({"source":str(r.path),"detections":detections},indent=2),encoding="utf-8")
    print(out.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    s = sub.add_parser("preview")
    s.add_argument("--image",required=True); s.add_argument("--json",required=True)
    s.add_argument("--out",default="preview.png"); s.set_defaults(func=preview)
    s = sub.add_parser("convert")
    s.add_argument("--source",required=True); s.add_argument("--out",default="dogface_yolo")
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--val-fraction",type=float,default=0.2); s.set_defaults(func=convert)
    s = sub.add_parser("train")
    s.add_argument("--data",default="dogface_yolo/data.yaml")
    s.add_argument("--model",default="yolo11n-pose.pt")
    s.add_argument("--epochs",type=int,default=1)
    s.add_argument("--imgsz",type=int,default=320)
    s.add_argument("--batch",type=int,default=4); s.set_defaults(func=train)
    s = sub.add_parser("predict")
    s.add_argument("--weights",required=True); s.add_argument("--source",required=True)
    s.add_argument("--out",default="predictions"); s.add_argument("--imgsz",type=int,default=320)
    s.add_argument("--conf",type=float,default=0.25); s.set_defaults(func=predict)
    a = p.parse_args()
    if a.command == "convert" and not 0 < a.val_fraction < 1:
        p.error("--val-fraction must be between 0 and 1")
    a.func(a)


if __name__ == "__main__":
    main()

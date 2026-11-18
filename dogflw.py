"""DogFLW: 犬の顔46点を変換・学習・評価し、写真とWebカメラで試す。"""
import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from contextlib import chdir
from pathlib import Path

from PIL import Image

POINT_COUNT = 46
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WORK = Path.home() / "Downloads" / "DogFLW_work"


def input_path(value, prompt):
    text = (value or input(prompt + "\n> ")).strip().strip('"')
    if not text:
        raise ValueError("フォルダーかファイルのパスを入力してください。")
    return Path(text).expanduser().resolve()


def dataset_root(folder):
    if not folder.is_dir():
        raise ValueError(f"展開したデータのフォルダーを指定してください。\n{folder}")
    if folder.name.lower() == "train":
        folder = folder.parent

    def usable(train):
        return all((train / part).is_dir() for part in ("images", "labels"))

    if usable(folder / "train"):
        return folder
    hits = [train.parent for train in folder.rglob("train") if usable(train)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ValueError("候補が複数あります。使うフォルダーを指定してください。\n"
                         + "\n".join(str(item) for item in hits))
    raise ValueError(f"{folder} の中に train/images と train/labels の組が見つかりません。"
                     "\nDogFLWの画像とJSONが入ったZIPを展開してください。")


def numbers(values, count, field):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{field}: {count} 個の値が必要")
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{field}: 空欄または数値以外") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field}: NaNまたは無限大")
    return values


def yolo_row(image, label):
    with Image.open(image) as img:
        width, height = img.size
        img.verify()
    data = json.loads(label.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("JSONはオブジェクト形式が必要")
    left, top, right, bottom = numbers(data.get("bounding_boxes"), 4, "顔の枠")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("顔の枠が画像の外、または幅・高さが不正")
    points = data.get("landmarks", data.get("labels"))
    if not isinstance(points, list) or len(points) != POINT_COUNT:
        raise ValueError(f"顔の点は {POINT_COUNT} 組必要")
    values = [(left + right) / (2 * width), (top + bottom) / (2 * height),
              (right - left) / width, (bottom - top) / height]
    for index, point in enumerate(points):
        x, y = numbers(point, 2, f"点 {index}")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"点 {index}: 画像の外 ({x}, {y})")
        values.extend((x / width, y / height))
    return "0 " + " ".join(f"{number:.8f}" for number in values) + "\n"


def collect(folder, hashes, skipped):
    images = folder / "images"
    labels = folder / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise ValueError(f"images と labels の両方を確認してください。\n{folder}")
    records = []
    used_stems = set()
    for image in sorted(images.iterdir()):
        if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
            continue
        label = labels / f"{image.stem}.json"
        try:
            row = yolo_row(image, label)
            with image.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest in hashes:
                raise ValueError(f"同一の画像ファイル: {hashes[digest]}")
            stem = image.stem.casefold()
            if stem in used_stems:
                raise ValueError("同名のTXTが既にあるため出力先が重複")
        except (ValueError, OSError, SyntaxError) as error:
            skipped.append({"split": folder.name, "image": image.name,
                            "label": label.name, "reason": str(error)})
            continue
        used_stems.add(stem)
        hashes[digest] = str(image)
        records.append((image, row))
    print(f"{folder.name}: 使用可能 {len(records)} 枚", flush=True)
    return records


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def convert(args):
    dataset = args.work / "dataset"
    if dataset.exists():
        raise ValueError(f"変換先が既にあります: {dataset}\n"
                         "変換が済んでいれば train に進んでください。\n"
                         "別に変換する場合はコマンドの前に --work で別の保存先を選べます。")
    root = dataset_root(input_path(args.source, "展開したDogFLWのフォルダーパスを貼り付けて Enter"))
    print(f"元データ: {root}", flush=True)
    hashes = {}
    skipped = []
    test = collect(root / "test", hashes, skipped) if (root / "test").is_dir() else []
    train = collect(root / "train", hashes, skipped)
    args.work.mkdir(parents=True, exist_ok=True)
    if len(train) < 2:
        write_json(args.work / "conversion_errors.json", skipped)
        raise ValueError(f"trainに有効な画像・JSONのペアが2組以上必要です。\n{args.work}")
    random.Random(42).shuffle(train)
    n_val = min(len(train) - 1, max(1, round(len(train) * 0.2)))
    splits = {"train": train[n_val:], "val": train[:n_val]}
    if test:
        splits["test"] = test
    with tempfile.TemporaryDirectory(dir=args.work) as temp:
        stage = Path(temp) / "dataset"
        for split, records in splits.items():
            image_dir = stage / "images" / split
            label_dir = stage / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for image, row in records:
                shutil.copyfile(image, image_dir / image.name)
                (label_dir / f"{image.stem}.txt").write_text(row, encoding="utf-8")
        config = ["path: " + json.dumps(dataset.as_posix(), ensure_ascii=False)]
        config.extend(f"{split}: images/{split}" for split in splits)
        config.extend(["kpt_shape: [46, 2]", "names:", "  0: dog_face"])
        (stage / "data.yaml").write_text("\n".join(config) + "\n", encoding="utf-8")
        write_json(stage / "skipped.json", skipped)
        stage.rename(dataset)
    print("  ".join(f"{split}={len(records)}" for split, records in splits.items()))
    print(f"除外 {len(skipped)} 件: {dataset / 'skipped.json'}\n設定: {dataset / 'data.yaml'}")
    if not test:
        print("使用可能なtest画像が0枚です。最終評価には別の評価用データが必要です。")


def train_model(args, yaml):
    from ultralytics import YOLO

    # 初期重みや学習結果の保存先を作業フォルダーにそろえる。
    with chdir(args.work):
        model = YOLO("yolo11n-pose.pt")
        model.train(data=str(yaml), device="cpu", epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=0, seed=42, fliplr=0.0, flipud=0.0,
                    mosaic=0.0, mixup=0.0, project=str(args.work / "runs"), name="train")
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"学習済みの重みを確認してください: {best}")
    dest = args.work / "best.pt"
    with tempfile.TemporaryDirectory(dir=args.work) as temp:
        copy = Path(temp) / "best.pt"
        shutil.copyfile(best, copy)
        copy.replace(dest)
    print(f"今回のモデル: {best}\n写真・カメラで使用するコピー: {dest}")


def predict_frame(model, image, args):
    return model.predict(source=image, device="cpu", imgsz=args.imgsz,
                         conf=args.conf, verbose=False, save=False)[0]


def photo(model, args):
    image = input_path(args.source, "試したい犬の写真のファイルパスを貼り付けて Enter")
    if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"写真ファイルを選んでください。\n{image}")
    result = predict_frame(model, str(image), args)
    faces = []
    if result.boxes is not None and result.keypoints is not None:
        for box, score, points in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                      result.keypoints.xy.tolist(), strict=True):
            faces.append({"box_xyxy": box, "face_confidence": score, "landmarks_xy": points})
    out = Path(tempfile.mkdtemp(prefix="photo-", dir=args.work))
    result.save(filename=str(out / "preview.jpg"), kpt_line=False, kpt_radius=2)
    write_json(out / "points.json", {"image": str(image), "faces": faces})
    print(f"検出した顔: {len(faces)}\n点を重ねた写真: {out / 'preview.jpg'}\n座標: {out / 'points.json'}")


def webcam(model, args):
    import cv2

    camera = cv2.VideoCapture(args.camera)
    title = "DogFLW | Q or Esc to close"
    try:
        if not camera.isOpened():
            raise RuntimeError("カメラを開けません。Windowsのカメラのアクセス許可と番号を確認してください。")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("カメラから映像を取得できませんでした。")
            result = predict_frame(model, frame, args)
            cv2.imshow(title, result.plot(kpt_line=False, kpt_radius=2))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK,
                        help="データと結果の保存先。サブコマンドの前に指定")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("convert", help="画像とJSONをYOLO形式にする")
    prep.add_argument("--source", help="DogFLWを展開したフォルダー。省略すると入力欄を表示")
    learn = sub.add_parser("train", help="CPUで学習")
    learn.add_argument("--epochs", type=int, default=10)
    evaluate = sub.add_parser("test", help="元のtestで最終評価")
    picture = sub.add_parser("photo", help="写真の顔の点を画像とJSONに出力")
    picture.add_argument("--source", help="写真のパス。省略すると入力欄を表示")
    camera = sub.add_parser("webcam", help="カメラに顔の点を表示")
    camera.add_argument("--camera", type=int, default=0)
    for cmd in (learn, evaluate, picture, camera):
        cmd.add_argument("--imgsz", type=int, default=320)
    for cmd in (learn, evaluate):
        cmd.add_argument("--batch", type=int, default=4)
    for cmd in (picture, camera):
        cmd.add_argument("--conf", type=float, default=0.25)
    for cmd in (evaluate, picture, camera):
        cmd.add_argument("--weights", type=Path, help="使用するモデル。初期値は作業フォルダーのbest.pt")
    args = parser.parse_args()
    args.work = args.work.expanduser().resolve()
    if args.command == "convert":
        convert(args)
        return
    if min(args.imgsz, getattr(args, "batch", 1), getattr(args, "epochs", 1)) <= 0:
        parser.error("imgsz, batch, epochs は正の整数で指定してください")
    if not 0 <= getattr(args, "conf", 0.25) <= 1:
        parser.error("conf は0～1の範囲で指定してください")
    yaml = args.work / "dataset" / "data.yaml"
    if args.command in ("train", "test") and not yaml.is_file():
        raise FileNotFoundError(f"先に convert を実行してください。\n{yaml}")
    if args.command == "train":
        train_model(args, yaml)
        return
    from ultralytics import YOLO

    weights = (args.weights or (args.work / "best.pt")).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"学習済みモデルが見つかりません。\n{weights}")
    model = YOLO(str(weights))
    if model.task != "pose":
        raise ValueError(f"Poseモデルを指定してください。\n{weights}")
    if args.command == "test":
        if not (yaml.parent / "images" / "test").is_dir():
            raise ValueError("有効なtest画像が変換済みデータにありません。")
        metrics = model.val(data=str(yaml), split="test", device="cpu", imgsz=args.imgsz,
                            batch=args.batch, workers=0, project=str(args.work / "runs"), name="test")
        print(f"test Pose mAP50-95 = {metrics.pose.map:.4f}")
    elif args.command == "photo":
        args.work.mkdir(parents=True, exist_ok=True)
        photo(model, args)
    elif args.command == "webcam":
        webcam(model, args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as problem:
        raise SystemExit(str(problem)) from None

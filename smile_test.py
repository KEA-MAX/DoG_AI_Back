import argparse
import json
import math
from pathlib import Path

# テスト用の仮のしきい値
MOUTH_MIN = 0.16
EYE_MIN = 0.45


def ratio(points, vertical, horizontal):
    width = math.dist(points[horizontal[0]], points[horizontal[1]])
    if width <= 0:
        raise ValueError("横幅が0のため比率を計算できません")
    return math.dist(points[vertical[0]], points[vertical[1]]) / width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8-sig"))

    if not data["faces"]:
        print("判定できません（犬の顔が見つかりませんでした）")
        return

    face = max(data["faces"], key=lambda face: face["face_confidence"])
    points = face["landmarks_xy"]

    if len(points) != 46:
        print(f"判定できません（46点が必要ですが、{len(points)}点でした）")
        return

    try:
        mouth = ratio(points, (38, 45), (39, 40))
        eye_a = ratio(points, (20, 22), (16, 18))
        eye_b = ratio(points, (21, 23), (17, 19))
    except ValueError as error:
        print(f"判定できません（{error}）")
        return

    smile_like = mouth >= MOUTH_MIN and min(eye_a, eye_b) >= EYE_MIN
    print(f"口元の比率 {mouth:.3f} / 目の比率 {eye_a:.3f}, {eye_b:.3f}")
    print("判定：笑顔っぽい" if smile_like else "判定：ふつう")


if __name__ == "__main__":
    main()
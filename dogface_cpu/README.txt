犬の顔46点・CPU用スターター（venv不要）

確認した入力: n02115913_423.json。landmarks は46組のピクセルXY、
bounding_boxes は文字列4個の [左, 上, 右, 下]。
このサンプルに合わせた変換です。全データの形式は未確認です。
異なる形式や範囲外の座標はエラーにし、黙って修正しません。

1. このZIPを展開し、そのフォルダーをVS Codeで開きます。
PowerShellのターミナルで実行してください。
既にUltralyticsがインストール済みなら追加インストールは不要です。

2. データセットの構成を確認します。
DogFLW/train/images/*.png （jpg等も対応）
DogFLW/train/labels/*.json
DogFLW/test/images/*.png （testはあればそのまま保持）
DogFLW/test/labels/*.json
imagesとlabels内は平坦な構成を想定しています。
以下のパスを、実際にtrainフォルダーが入っている場所に変更します。

py dogface.py convert --source "C:\datasets\DogFLW"

元画像は変更しません。dogface_yoloにコピーしてYOLO形式に変換します。
元のtrainの80%を学習、20%を検証に使用。testは学習に使いません。
画像の完全一致はエラーで停止します。同じ犬の別写真や似た画像は
自動判別できません。個体IDがあれば個体単位で分割する必要があります。
出力先が存在する場合は --out dogface_yolo_v2 等で別名にしてください。
この場合、以降の --data もそのフォルダーのdata.yamlに変更します。

3. CPUでまず動作確認（1周だけ）:
py dogface.py train --epochs 1

初回は学習済みyolo11n-pose.ptのダウンロードにネット接続が必要です。
これは人の姿勢モデルを初期値にして犬の顔46点へ追加学習する処理です。
device=cpu、workers=0、batch=4、imgsz=320。
左右対称点の正式対応を未確認のため左右反転は無効です。
1周は動作確認専用で、精度を期待する設定ではありません。
CPU時間は実機で確認してください。GPUなしでも学習は可能です。

4. 動作したら学習の試行例:
py dogface.py train --epochs 50 --imgsz 640 --batch 2

50周で十分という保証はありません。320より640は計算負荷が高くなります。
小さな顔の詳細には画像解像度も影響します。まず1周の時間を確認します。
保存先は端末の表示を確認。繰り返すとtrain2等になる場合があります。

5. 学習後、新しい画像の点を出力:
py dogface.py predict --weights "runs\dogface\train\weights\best.pt" --source "C:\photos\dog.jpg" --imgsz 640

weightsは実際の保存先に変更します。predictionsに画像と座標JSONを保存。
再実行は --out predictions2 等で新しいフォルダーを指定します。
座標は元画像のピクセル単位。点番号は入力配列の0～45番の順番です。
detection_confidenceは顔検出のスコアで、笑顔の確率ではありません。
点の正式な解剖学的名称は、このコードでは推測して割り当てていません。

研究上の注意:
これは顔の点を学習するコードです。嬉しさや笑顔は判定しません。
可視性ラベルがないので kpt_shape: [46,2] を使います。
笑顔らしい外見の分類には、定義と独立した正解ラベルが別途必要です。
精度は未使用画像で測定し、犬種・顔の角度・遮蔽でも分けて確認します。
標準のpose指標だけで感情推定の正確さは評価できません。

検証範囲:
提供サンプルの座標変換・プレビュー、および小規模な変換試験を実施。
この環境ではYOLOの実学習・推論は未実行です。

資料:
https://github.com/martvelge/DogFLW
https://www.nature.com/articles/s41598-025-07040-3
https://docs.ultralytics.com/datasets/pose/
https://docs.ultralytics.com/modes/train/
データセットのライセンスは元の配布条件を確認してください。
DogFLWリポジトリーではCC BY-NC 4.0（帰属表示・非営利）が案内されています。

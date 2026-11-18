# DoG_AI_Back



別PCのVS Codeで DoG_AI_Back を開いて、PowerShellでこれを実行してね。学習済みの300エポック版をそのまま使えるよ。
git lfs install
git pull
git lfs pull


py -3.14 -m pip install mss
$scripts = py -3.14 -c "import sysconfig; print(sysconfig.get_path('scripts'))"

& "$scripts\yolo.exe" pose predict "model=.\models\dogface-4090-300.pt" "source=screen 1 0 0 960 720" imgsz=640 show=True save=False
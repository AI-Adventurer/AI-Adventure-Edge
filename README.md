# AI Adventure Edge

這是把原本 ROS 版體感冒險遊戲移植成非 ROS、可直接在 Jetson 上執行的版本。

目前專案包含：

- OpenCV 相機輸入與遊戲 UI
- MediaPipe Pose 骨架擷取
- CTR-GCN 動作辨識
- 純 Python 遊戲狀態機
- Jetson 可用的 PyTorch GPU / TensorRT 工作流

## 名稱說明

目前專案資料夾名稱是 `AI_Adventure_Edge`，但 Python 套件名稱仍然是
`adventure_game_jetson`，所以啟動指令還是：

```bash
python -m adventure_game_jetson.app
```

安裝後的 console script 也是：

```bash
adventure-game
```

## ROS 對應說明

如果你原本是在找 ROS 版的 `game_manager_node`，這一版的核心邏輯已經拆成純 Python 類別，
主要放在 [src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py) 的
`GameEngine`。

大致對應關係如下：

- 原本的 `game_manager` 狀態機與流程控制 -> [src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py)
- 劇情模板、事件資料、OpenAI 劇情生成 -> [src/adventure_game_jetson/core/story.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/story.py)
- 主程式串接相機、辨識、遊戲狀態、UI -> [src/adventure_game_jetson/app/main.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/app/main.py)

如果你想修改遊戲流程、關卡切換、成功失敗判定、血量分數規則，優先從
[src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py)
開始看。

## 專案結構

- `models/`
  - `config.yaml`: CTR-GCN 模型設定
  - `best.pt`: PyTorch 權重
  - `ctrgcn_fp16.engine`: TensorRT engine
- `scripts/`
  - `install_jetson_cusparselt.sh`: 安裝 cuSPARSELt
  - `install_jetson_torch.sh`: 安裝 Jetson GPU torch wheel
  - `build_ctrgcn_engine.sh`: 從 `best.pt` 重建 TensorRT engine
- `src/adventure_game_jetson/app/`
  - CLI 入口與主迴圈
- `src/adventure_game_jetson/capture/`
  - 相機 / 影片來源，支援 GStreamer 與一般 webcam
- `src/adventure_game_jetson/core/`
  - 遊戲狀態機、劇情、事件資料
- `src/adventure_game_jetson/inference/`
  - MediaPipe Pose、CTR-GCN、PyTorch / TensorRT backend、profiling
- `src/adventure_game_jetson/ui/`
  - OpenCV HUD、中文文字繪製、骨架 overlay

## 執行流程

整體資料流如下：

1. `VideoSource` 讀取相機或影片 frame
2. `ActionRecognizer` 用 MediaPipe 擷取 33 點骨架
3. 骨架經過前處理後送進 CTR-GCN
4. `GameEngine` 根據辨識結果推進遊戲狀態
5. `GameRenderer` 把相機畫面、骨架、劇情、倒數、動作結果畫出來

關鍵檔案：

- [src/adventure_game_jetson/app/main.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/app/main.py)
- [src/adventure_game_jetson/inference/runtime.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/inference/runtime.py)
- [src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py)
- [src/adventure_game_jetson/ui/renderer.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/ui/renderer.py)

## Jetson 安裝建議

建議環境：

- Jetson Orin Nano / Orin 系列
- Python 3.10
- conda 或 mamba

注意：

- Jetson 的 GPU torch 請不要用 generic `pip install torch`
- 先安裝 `cuSPARSELt`，再安裝 NVIDIA 提供的 Jetson torch wheel
- 這個 repo 的 torch 安裝腳本目前明確要求 Python 3.10

### 方式 A: conda

```bash
cd /home/jetson/workspace/AI_Adventure_Edge
conda env create -f environment.yml
conda activate adventure_game_jetson
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
pip install -e .
./scripts/install_jetson_cusparselt.sh
./scripts/install_jetson_torch.sh
```

如果你想用 OpenAI 劇情功能：

```bash
pip install ".[openai]"
```

### 方式 B: venv

```bash
cd /home/jetson/workspace/AI_Adventure_Edge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
./scripts/install_jetson_cusparselt.sh
./scripts/install_jetson_torch.sh
```

## 啟動方式

### 最基本執行

```bash
cd /home/jetson/workspace/AI_Adventure_Edge
python -m adventure_game_jetson.app
```

### 指定字型，避免中文字缺字

```bash
python -m adventure_game_jetson.app \
  --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

### Jetson 比較推薦的效能設定

```bash
python -m adventure_game_jetson.app \
  --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
  --profile \
  --mp-model-complexity 0 \
  --mp-input-width 256 \
  --mp-input-height 256 \
  --pose-every-n-frames 2 \
  --window-width 1280 \
  --window-height 720
```

### 使用 TensorRT 的 CTR-GCN

```bash
python -m adventure_game_jetson.app \
  --action-backend tensorrt \
  --action-engine /home/jetson/workspace/AI_Adventure_Edge/models/ctrgcn_fp16.engine
```

### 直接用 console script

```bash
adventure-game
```

結束方式：

- 按 `q`
- 或按 `Esc`

## TensorRT 與 MediaPipe 說明

### 動作辨識 backend

程式預設不是固定用 `pytorch`，而是 `--action-backend auto`。

在 `auto` 模式下：

- 如果裝置是 CUDA，且有可用的 TensorRT engine，會優先走 TensorRT
- 否則退回 PyTorch

相關程式在：

- [src/adventure_game_jetson/inference/backends/__init__.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/inference/backends/__init__.py)
- [src/adventure_game_jetson/inference/backends/tensorrt_ctrgcn.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/inference/backends/tensorrt_ctrgcn.py)

### Pose backend

目前 pose backend 只有 `mediapipe`。

可以用這些參數調整效能：

- `--mp-model-complexity`
- `--mp-input-width`
- `--mp-input-height`
- `--pose-every-n-frames`

如果 FPS 不夠，最先調這幾個。

### 重建 TensorRT engine

```bash
cd /home/jetson/workspace/AI_Adventure_Edge
./scripts/build_ctrgcn_engine.sh
```

如果你改了模型權重、模型設定，或改了輸入 shape，通常就要重建 engine。

## 遊戲規則與可修改位置

### 想改劇情

看：

- [src/adventure_game_jetson/core/story.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/story.py)

裡面有：

- `GAME_LOOPS`
- `EVENTS_DB`
- `GAME_END_1`
- `GAME_END_2`

### 想改成功失敗判定、分數、HP、狀態切換

看：

- [src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py)

### 想改相機或 GStreamer

看：

- [src/adventure_game_jetson/capture/video_source.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/capture/video_source.py)

### 想改 UI 樣式

看：

- [src/adventure_game_jetson/ui/renderer.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/ui/renderer.py)

## 疑難排解

### `git push` 出現 `src refspec main does not match any`

這代表 repo 還沒有任何 commit。你要先：

```bash
git add .
git commit -m "first commit"
git push -u origin main
```

### `conda: command not found`

先執行：

```bash
source ~/.bashrc
```

如果還不行：

```bash
source /home/jetson/miniconda3/etc/profile.d/conda.sh
```

### `torch.cuda.is_available()` 是 `False`

- 確認先跑過 `./scripts/install_jetson_cusparselt.sh`
- 確認用的是 Jetson 的 GPU torch wheel
- 確認 Python 版本是 3.10

### 中文字變方框或缺字

建議加上：

```bash
--font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

### OpenCV 視窗上下被吃掉

新版 renderer 已經盡量關掉 Qt GUI 工具列並改成縮放版面；如果你的桌面環境還是擠壓視窗，
可以把視窗再縮一點：

```bash
python -m adventure_game_jetson.app --window-width 1200 --window-height 680
```

### `libstdc++.so.6` / `CXXABI_*` 錯誤

在 conda 環境內執行：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

## 驗證環境

```bash
python - <<'PY'
import cv2, numpy, yaml, PIL, mediapipe, torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

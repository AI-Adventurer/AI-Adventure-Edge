# AI Adventure Edge

這個 repo 現在有兩種使用方式：

- `edge` 模式：Jetson Nano 只負責相機讀取、MediaPipe Pose、CTR-GCN 動作辨識，然後把骨架 / 動作 / 可選預覽影像送到遠端主機
- `standalone` 模式：保留原本單機版流程，讓 Jetson 本地直接跑 `GameEngine` 與 OpenCV 視窗，方便除錯或 demo

如果你的正式部署是「Jetson 做辨識，另一台電腦做 game manager 與 UI」，請優先使用 `edge` 模式。

如果遠端主機是 `/home/jetson/workspace/AI-Adventurer-APP-main` 這個後端，Jetson 端目前應該對接：

- Socket.IO path: `socket.io`
- namespace: `/edge/frames`
- event: `frame`
- 建議 transports: `polling,websocket`

如果你要把即時畫面也送到遠端的 `/edge/video`：

- Socket.IO path: `socket.io`
- namespace: `/edge/video`
- signaling events: `offer` / `answer` / `candidate`
- Jetson 端現在已支援用 WebRTC sender 透過這組 signaling 送出 video track

另外要注意：

- 這個遠端後端目前真正 ingest 的核心欄位是 `timestamp`、`source`、`frame_id`、`action_scores`、`stable_action`、`confidence`、`skeleton_sequence`
- Jetson 這邊額外送出的 `pose`、`timings_ms`、`runtime`、`preview_frame` 目前不會讓它壞掉，但也還沒有被那個後端的 game logic / frontend 正式消費
- `preview_frame` 是附在 frame payload 裡的縮圖；如果你要正式即時畫面，請改走 `--edge-video-url ...` 的 WebRTC 視訊通道

目前專案包含：

- OpenCV 相機輸入
- MediaPipe Pose 骨架擷取
- CTR-GCN 動作辨識
- Edge packet 輸出（NDJSON / optional Socket.IO）
- WebRTC video sender（optional）
- 純 Python 遊戲狀態機與本地 OpenCV UI（僅 `standalone` 模式使用）
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

## 建議部署架構

正式部署建議拆成兩台機器：

```text
Jetson Nano
  ├─ Camera capture
  ├─ MediaPipe Pose
  ├─ CTR-GCN inference
  └─ Edge packet output
         ↓
Remote Host
  ├─ Game Manager / 狀態機
  ├─ UI / 視窗顯示
  ├─ 規則判定
  └─ 狀態同步 / 後端服務
```

責任分工：

- Jetson Nano：相機讀取、骨架擷取、CTR-GCN 推論、輸出 raw pose / action scores / optional preview image
- Remote Host：遊戲規則、劇情、視窗 UI、玩家狀態、網頁或桌面端畫面

如果你要看傳輸 payload，請直接看 [Jetson-Nano-API-Documentation.md](/home/jetson/workspace/AI_Adventure_Edge/Jetson-Nano-API-Documentation.md)。

## 執行模式

### 1. `edge`

這是目前推薦的模式。

- 不建立本地 `GameEngine`
- 不開 Jetson 本地 OpenCV 視窗
- 每幀輸出：
  - `pose`
  - `skeleton_sequence`
  - `action_scores`
  - `stable_action`
  - optional `preview_frame`

### 2. `standalone`

這是保留給本地除錯 / demo 的模式。

- Jetson 本地跑 `GameEngine`
- Jetson 本地開 OpenCV 視窗
- 用來驗證辨識和 UI 疊圖是否正常

## ROS 對應說明

如果你原本是在找 ROS 版的 `game_manager_node`，這一版的核心邏輯已經拆成純 Python 類別，
主要放在 [src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py) 的
`GameEngine`。

注意：`GameEngine` 現在只在 `standalone` 模式需要；如果你走 edge 部署，game manager 應該放到遠端主機。

大致對應關係如下：

- 原本的 `game_manager` 狀態機與流程控制 -> [src/adventure_game_jetson/core/engine.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/engine.py)
- 劇情模板、事件資料、OpenAI 劇情生成 -> [src/adventure_game_jetson/core/story.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/core/story.py)
- 主程式串接相機、辨識、edge packet / 遊戲狀態 / UI -> [src/adventure_game_jetson/app/main.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/app/main.py)

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
  - CLI 入口與主迴圈（`standalone` / `edge`）
- `src/adventure_game_jetson/capture/`
  - 相機 / 影片來源，支援 GStreamer 與一般 webcam
- `src/adventure_game_jetson/core/`
  - 遊戲狀態機、劇情、事件資料（`standalone` 用）
- `src/adventure_game_jetson/edge/`
  - Edge packet builder、NDJSON / Socket.IO publisher、WebRTC video sender
- `src/adventure_game_jetson/inference/`
  - MediaPipe Pose、CTR-GCN、PyTorch / TensorRT backend、profiling
- `src/adventure_game_jetson/ui/`
  - OpenCV HUD、中文文字繪製、骨架 overlay（`standalone` 用）

## 執行流程

整體資料流如下：

1. `VideoSource` 讀取相機或影片 frame
2. `ActionRecognizer` 用 MediaPipe 擷取 33 點骨架
3. 骨架經過前處理後送進 CTR-GCN
4. `edge` 模式：封裝成 edge packet，輸出到 stdout / file / Socket.IO
5. `standalone` 模式：`GameEngine` 根據辨識結果推進遊戲狀態
6. `standalone` 模式：`GameRenderer` 把相機畫面、骨架、劇情、倒數、動作結果畫出來

關鍵檔案：

- [src/adventure_game_jetson/app/main.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/app/main.py)
- [src/adventure_game_jetson/edge/payloads.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/edge/payloads.py)
- [src/adventure_game_jetson/edge/publishers.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/edge/publishers.py)
- [src/adventure_game_jetson/edge/video.py](/home/jetson/workspace/AI_Adventure_Edge/src/adventure_game_jetson/edge/video.py)
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

如果你想讓 edge mode 直接把 JSON packet 發到 Socket.IO：

```bash
pip install ".[edge]"
```

如果你想讓 edge mode 再加上 WebRTC video：

```bash
pip install ".[edge,edge-video]"
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

### Edge mode: 輸出到 stdout（NDJSON）

```bash
cd /home/jetson/workspace/AI_Adventure_Edge
python -m adventure_game_jetson.app --mode edge
```

這個模式不開本地視窗，預設把 JSON packet 逐行寫到 stdout。

### Edge mode: 輸出到檔案

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --edge-output-path /tmp/jetson_edge.jsonl
```

### Edge mode: 送到遠端 Socket.IO

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.50.174:8000 \
  --edge-sio-namespace /edge/frames \
  --edge-sio-event frame \
  --edge-sio-transports polling,websocket
```

這組參數對應遠端後端的 `@socketio.on("frame", namespace="/edge/frames")`。

如果你想直接套用 Jetson 比較平衡的效能設定，也可以改成：

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --perf-mode balanced
```

`--perf-mode balanced` 目前會自動套用：

- `--width 640`
- `--height 480`
- `--mp-input-width 256`
- `--mp-input-height 192`
- `--pose-every-n-frames 2`
- `--stride 2`
- `--smooth-k 3`
- `--edge-publish-history-size 8`
- `--edge-video-fps 12`
- `--edge-video-width 640`
- `--edge-video-height 480`

如果你另外手動指定其中某個參數，手動值會優先，不會被 preset 蓋掉。

如果你想直接看 `edge` 模式的 FPS / 分段耗時，也可以加：

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --perf-mode balanced \
  --profile \
  --profile-every 60
```

這會定期輸出 `capture / pose / preprocess / action / total / fps`，比較容易知道瓶頸是在相機、MediaPipe、CTR-GCN 還是整體主迴圈。

如果後端用了不同 namespace，再改這個值：

```bash
--edge-sio-namespace /你的namespace
```

如果後端 event 名稱不是 `frame`，請改成後端 `@socketio.on(...)` 的事件名稱。

### Edge mode: 同時夾帶 UI 用預覽影像

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.50.174:8000 \
  --edge-sio-namespace /edge/frames \
  --edge-sio-event frame \
  --edge-sio-transports polling,websocket \
  --edge-include-preview \
  --edge-preview-width 640 \
  --edge-preview-height 360 \
  --edge-preview-every-n-frames 3 \
  --edge-preview-overlay
```

這個做法是把縮圖塞進 `frame` payload，適合除錯或暫時過渡，不是正式的即時視訊通道。

### Edge mode: 同時送骨架結果 + WebRTC video

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.50.174:8000 \
  --edge-sio-namespace /edge/frames \
  --edge-sio-event frame \
  --edge-sio-transports polling,websocket \
  --edge-video-url http://192.168.50.174:8000 \
  --edge-video-namespace /edge/video \
  --edge-video-path socket.io \
  --edge-video-transports polling,websocket \
  --edge-video-offer-event offer \
  --edge-video-answer-event answer \
  --edge-video-candidate-event candidate \
  --edge-video-width 640 \
  --edge-video-height 360
```

重點：

- `--edge-video-url` 一設下去，就會啟動 Jetson 端的 WebRTC sender
- 這條 video stream 會跟 CTR-GCN 共用同一支 camera，不會另外搶一次 `VideoCapture`
- 如果遠端需要 STUN / TURN，可以用 `--edge-video-ice-servers` 傳逗號分隔的 server URL

### Standalone mode: 最基本執行

```bash
python -m adventure_game_jetson.app \
  --mode standalone \
  --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

### Standalone mode: Jetson 比較推薦的效能設定

```bash
python -m adventure_game_jetson.app \
  --mode standalone \
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
  --mode standalone \
  --action-backend tensorrt \
  --action-engine /home/jetson/workspace/AI_Adventure_Edge/models/ctrgcn_fp16.engine
```

### 直接用 console script

```bash
adventure-game
```

結束方式：

- `edge` 模式：`Ctrl+C`
- `standalone` 模式：按 `q` 或 `Esc`

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

# Jetson Nano API Documentation

本文件描述目前這個 repo 在 `--mode edge` 下的實際輸出介面。

在新的部署模型中：

- Jetson Nano：只負責相機讀取、MediaPipe Pose、CTR-GCN 推論、輸出資料
- Remote Host：負責 game manager、規則判定、UI / 視窗、狀態同步

## 架構

```text
Jetson Nano
  ├─ Camera capture
  ├─ MediaPipe Pose
  ├─ CTR-GCN inference
  ├─ Edge packet output
  │      └─ Socket.IO / NDJSON
  └─ Edge video output
         └─ WebRTC media + Socket.IO signaling
                ↓
Remote Host
  ├─ Game Manager
  ├─ UI / Window Renderer
  ├─ Rule Engine
  └─ Backend Services
```

## 傳輸方式

目前程式支援兩種輸出方式：

1. NDJSON
   - `--edge-output-path /path/to/file.jsonl`
   - 如果沒有指定 `--edge-sio-url`，預設直接輸出到 stdout
2. Socket.IO
   - `--edge-sio-url http://<remote-host>:8000`
   - `--edge-sio-path socket.io`
   - `--edge-sio-namespace /edge/frames`
   - `--edge-sio-event frame`
   - `--edge-sio-transports polling,websocket`
   - 需要安裝 `python-socketio[client]`
3. WebRTC Video
   - `--edge-video-url http://<remote-host>:8000`
   - `--edge-video-namespace /edge/video`
   - `--edge-video-offer-event offer`
   - `--edge-video-answer-event answer`
   - `--edge-video-candidate-event candidate`
   - 需要安裝 `aiortc`

注意：

- `localhost` 不應該寫死；Remote Host 在另一台機器時，請填遠端主機的 IP 或 hostname
- `preview_frame` 是 frame payload 內的 optional 縮圖，不等於正式 WebRTC 視訊
- 如果遠端主機是 `/home/jetson/workspace/AI-Adventurer-APP-main`，目前後端實作對應的是 Socket.IO namespace `/edge/frames` 上的 `frame` 事件；`POST /edge/frames` 只是備援 HTTP endpoint
- 如果遠端主機已經把 `/edge/video` 處理好，Jetson 端可以直接用 `--edge-video-*` 啟動 WebRTC sender

## 1. Socket.IO

### Purpose

Jetson Nano 連到 Remote Host 的 Socket.IO server，並用 event 送出每一幀的骨架辨識結果與 optional 預覽影像。

### Connection

- Base URL: `http://<remote-host>:8000`
- Socket.IO Path: `socket.io`
- Namespace: 預設 `/edge/frames`
- Event: 預設 `frame`

### CLI Example

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.1.50:8000 \
  --edge-sio-namespace /edge/frames \
  --edge-sio-event frame \
  --edge-sio-transports polling,websocket
```

如果後端 namespace 不是 `/edge/frames`：

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.1.50:8000 \
  --edge-sio-event frame \
  --edge-sio-namespace /你的namespace
```

`frame` 只是目前和 `AI-Adventurer-APP-main` 對接時的預設值；實際上仍要以後端 `@socketio.on(...)` 定義為準。

## 2. WebRTC Video

### Purpose

Jetson Nano 把同一支 camera 的畫面另外做成 WebRTC video track，提供遠端 UI / browser 顯示即時畫面。

### Signaling

- Base URL: `http://<remote-host>:8000`
- Socket.IO Path: `socket.io`
- Namespace: 預設 `/edge/video`
- Offer Event: 預設 `offer`
- Answer Event: 預設 `answer`
- Candidate Event: 預設 `candidate`

### CLI Example

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
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

### Notes

- Jetson 端現在會讓 WebRTC video 和 CTR-GCN inference 共用同一支 camera
- 如果需要 STUN / TURN，可以用 `--edge-video-ice-servers` 傳逗號分隔的 URL
- `preview_frame` 只適合除錯；正式即時畫面建議走這條 WebRTC channel

## 3. Payload Format

```json
{
  "timestamp": 1712345678.12,
  "source": "jetson-nano-01",
  "frame_id": 1523,
  "prediction_ready": true,
  "action_scores": {
    "stand": 0.02,
    "jump": 0.03,
    "crouch": 0.82,
    "push": 0.08,
    "run_forward": 0.05
  },
  "stable_action": "crouch",
  "confidence": 0.82,
  "frame": {
    "width": 640,
    "height": 480
  },
  "pose": {
    "layout": "mediapipe_pose_33",
    "shape": [33, 3],
    "points": [
      [0.512, 0.103, -0.021],
      [0.438, 0.221, -0.034],
      [0.587, 0.219, -0.031]
    ]
  },
  "skeleton_sequence": {
    "layout": "mediapipe_pose_33",
    "shape": [30, 33, 3],
    "frames": [
      [
        [0.512, 0.103, -0.021],
        [0.438, 0.221, -0.034],
        [0.587, 0.219, -0.031]
      ],
      [
        [0.511, 0.105, -0.022],
        [0.439, 0.224, -0.035],
        [0.588, 0.221, -0.032]
      ]
    ]
  },
  "preview_frame": {
    "encoding": "jpeg_base64",
    "width": 640,
    "height": 360,
    "overlay": "skeleton",
    "data": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
  },
  "timings_ms": {
    "capture": 7.4,
    "pose": 18.2,
    "preprocess": 0.3,
    "action": 2.4,
    "total": 28.6
  },
  "runtime": {
    "pose_backend": "mediapipe",
    "action_backend": "tensorrt",
    "action_device": "cuda"
  }
}
```

### 必要欄位

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `timestamp` | float | Unix timestamp，使用 `time.time()`，可跨機器對時 |
| `source` | string | Jetson 裝置識別字串 |
| `frame_id` | int | 幀序號，從 1 遞增 |
| `prediction_ready` | bool | CTR-GCN 視窗是否已暖機完成 |
| `action_scores` | object | 五個 action label 的 softmax 分數 |
| `stable_action` | string | 目前穩定輸出的動作類別 |
| `confidence` | float | `stable_action` 對應的信心度 |
| `frame` | object | 原始相機 frame 尺寸 |
| `pose` | object | 目前這一幀的 raw pose |
| `skeleton_sequence` | object | 最近一段時間的 raw pose 序列 |
| `timings_ms` | object | 捕捉 / pose / action 推論耗時 |
| `runtime` | object | 當前 backend 與 device 資訊 |

### action_scores 類別

目前專案固定輸出這五個 label：

- `stand`
- `jump`
- `crouch`
- `push`
- `run_forward`

### pose 欄位

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `layout` | string | 目前固定為 `mediapipe_pose_33` |
| `shape` | array | `[33, 3]` |
| `points` | array | 33 個關鍵點，每個點是 `[x, y, z]` |

`pose.points` 為 raw pose：

- 尚未做 CTR-GCN 用的 `centralize`
- 尚未做模型輸入重取樣或插值
- 這份資料適合遠端 UI 疊骨架、記錄、除錯

### skeleton_sequence 欄位

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `layout` | string | 目前固定為 `mediapipe_pose_33` |
| `shape` | array | `[T, 33, 3]`，其中 `T <= window_size` |
| `frames` | array | 最近 T 幀的 raw pose |

注意：

- 預設 `window_size` 是 30，所以常見 shape 會是 `[30, 33, 3]`
- 如果 `pose_every_n_frames > 1`，中間未重新跑 pose 的 frame 會重用最近一次骨架

### preview_frame 欄位

`preview_frame` 是 optional，只有在加上 `--edge-include-preview` 時才會出現。

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `encoding` | string | 目前固定為 `jpeg_base64` |
| `width` | int | 預覽圖寬 |
| `height` | int | 預覽圖高 |
| `overlay` | string | `none` 或 `skeleton` |
| `data` | string | JPEG base64 字串 |

建議：

- UI 只需要骨架與動作時，可關閉 `preview_frame` 以節省頻寬
- 如果需要畫面預覽，建議先用 `640x360` 搭配 `--edge-preview-every-n-frames 3`
- 如果遠端是 `AI-Adventurer-APP-main` 目前那版後端，`preview_frame` 會被視為額外欄位而忽略；它現在真正 ingest 的仍是核心 frame payload

### Success

- Socket.IO 連線建立後，Jetson 持續以 event 送出 frame packet

### Optional Server Acknowledgment

如果後端要回 ACK，可以採用以下格式：

```json
{
  "success": true,
  "frame_id": 1523,
  "processed_at": 1712345678.15
}
```

目前這個 repo 的 Jetson client 不依賴 ACK；是否回覆由 server 決定。

### Errors

- `Invalid JSON`: payload 不是有效 JSON
- `Missing required field`: 缺少必要欄位
- `Invalid action_scores`: 分數物件缺欄位或值不在 0.0-1.0
- `Invalid pose shape`: `pose.shape` 和 `pose.points` 不一致
- `Invalid skeleton_sequence shape`: `shape` 和 `frames` 維度不一致
- `Frame too large`: `preview_frame` 過大
- `Connection timeout`: 長時間沒有收到資料

## 4. CLI Examples

### 輸出到 stdout

```bash
python -m adventure_game_jetson.app \
  --mode edge
```

### 輸出到 JSONL 檔案

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --edge-output-path /tmp/jetson_edge.jsonl
```

### 輸出到遠端 Socket.IO

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.1.50:8000 \
  --edge-sio-namespace /edge/frames \
  --edge-sio-event frame \
  --edge-sio-transports polling,websocket
```

### 帶預覽圖一起送

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

### 同時送 frame packet + WebRTC video

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

## 5. Integration Notes

遠端主機如果要做 UI / game manager，通常只需要這些欄位：

- `stable_action`
- `confidence`
- `pose.points`
- `preview_frame`（如果 UI 要顯示即時畫面）

如果遠端主機是 `AI-Adventurer-APP-main` 的目前版本，實際狀況再補充兩點：

- 它目前真正驗證與存下來的是 `timestamp`、`source`、`frame_id`、`action_scores`、`stable_action`、`confidence`、`skeleton_sequence`
- `preview_frame` 在那個後端目前沒有正式 consumer；正式即時畫面應該走 `/edge/video` 那條 WebRTC signaling 流程

建議責任切分：

- Jetson：提供感知結果，不做遊戲規則
- Remote Host：消費 Jetson 資料，決定下一步事件、分數、血量、畫面更新

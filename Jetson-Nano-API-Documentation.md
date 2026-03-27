# Jetson Nano API Documentation

本文件描述目前 Jetson Nano 在 `--mode edge` 下，實際送給遠端主機的資料格式。

目前專案的責任切分是：

- Jetson Nano：相機讀取、MediaPipe Pose、CTR-GCN 推論、送出骨架 / 動作 / 視訊
- Remote Host：遊戲規則、狀態管理、UI / browser 顯示

## 架構

```text
Jetson Nano
  ├─ Camera capture
  ├─ MediaPipe Pose
  ├─ CTR-GCN inference
  ├─ Edge frame packet
  │      └─ Socket.IO / NDJSON
  └─ Edge video preview
         └─ WebRTC media + Socket.IO signaling
                ↓
Remote Host
  ├─ Backend (/edge/frames, /edge/video)
  ├─ Game Manager
  └─ UI / Browser
```

## 1. Frame Transport

### Socket.IO

- Base URL: `http://<remote-host>:8000`
- Path: `socket.io`
- Namespace: 預設 `/edge/frames`
- Event: 預設 `frame`

CLI 範例：

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.50.174:8000 \
  --edge-sio-namespace /edge/frames \
  --edge-sio-event frame \
  --edge-sio-transports polling,websocket
```

### NDJSON

如果沒有指定 `--edge-sio-url`，frame packet 會輸出到 stdout；也可以用：

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --edge-output-path /tmp/jetson_edge.jsonl
```

## 2. Frame Payload

Jetson 每一幀送出的 packet 格式如下：

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

### 重要說明

- `pose.points` 和 `skeleton_sequence.frames` 目前都是 3 維座標 `[x, y, z]`
- 目前沒有傳 `visibility`
- `pose` 是單幀 raw pose
- `skeleton_sequence` 是最近 `T` 幀的 raw pose 序列
- `T` 通常等於 `window_size`，預設是 30

### 必要欄位

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `timestamp` | float | Unix timestamp |
| `source` | string | Jetson 裝置識別字串 |
| `frame_id` | int | 幀序號 |
| `action_scores` | object | 動作分類分數 |
| `stable_action` | string | 目前穩定輸出的動作 |
| `confidence` | float | `stable_action` 信心度 |
| `skeleton_sequence` | object | 最近一段骨架序列 |

### pose 欄位

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `layout` | string | 目前固定 `mediapipe_pose_33` |
| `shape` | array | 固定 `[33, 3]` |
| `points` | array | 33 個點，每點為 `[x, y, z]` |

補充：

- `x`、`y` 是 MediaPipe 標準化座標
- `z` 是 MediaPipe 相對深度
- 若某幀沒有抓到 pose，目前會回傳 33 個全 0 點，而不是空陣列

### skeleton_sequence 欄位

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `layout` | string | 目前固定 `mediapipe_pose_33` |
| `shape` | array | `[T, 33, 3]` |
| `frames` | array | 最近 T 幀，每幀為 33 個 `[x, y, z]` |

補充：

- `skeleton_sequence` 保存的是 raw pose history，不是 centralize 後的 CTR-GCN 輸入
- 如果 `pose_every_n_frames > 1`，中間未重跑 pose 的 frame 會沿用前一次骨架
- 如果遠端後端只存 `skeleton_sequence`，要拿最新單幀骨架可以用 `frames[-1]`

### action labels

目前固定這五類：

- `stand`
- `jump`
- `crouch`
- `push`
- `run_forward`

## 3. Preview Frame

若有加 `--edge-include-preview`，payload 會多一個 optional `preview_frame`：

```json
{
  "preview_frame": {
    "encoding": "jpeg_base64",
    "width": 640,
    "height": 360,
    "overlay": "skeleton",
    "data": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }
}
```

注意：

- 這只是縮圖預覽，不是正式視訊流
- 遠端 `AI-Adventurer-APP-main` 目前 ingest / store 的核心欄位不包含 `preview_frame`

## 4. WebRTC Video

Jetson 也可以把相機畫面走 WebRTC 傳給遠端 UI。

- Base URL: `http://<remote-host>:8000`
- Socket.IO path: `socket.io`
- Namespace: 預設 `/edge/video`
- Offer event: `offer`
- Answer event: `answer`
- Candidate event: `candidate`

CLI 範例：

```bash
python -m adventure_game_jetson.app \
  --mode edge \
  --source-id jetson-nano-01 \
  --edge-sio-url http://192.168.50.174:8000 \
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

## 5. 與遠端後端對接時要注意

如果遠端是 `/home/jetson/workspace/AI-Adventurer-APP-main` 目前這版：

- live Socket.IO `frame` event 會收到完整 payload，包含 `pose`、`timings_ms`、`runtime`
- backend 驗證必要欄位時，真正必填的是 `timestamp`、`source`、`frame_id`、`action_scores`、`stable_action`、`confidence`、`skeleton_sequence`
- backend 存進 `JetsonFrame` 時，目前只保留核心欄位和 `skeleton_sequence`
- 也就是說，如果你看 `GET /edge/frames/latest/<source>`，不一定會看到 `pose` 或 `preview_frame`

## 6. 結論

目前骨架資料的正確理解是：

- 單幀骨架：`pose.points`
- 序列骨架：`skeleton_sequence.frames`
- 座標維度：`[x, y, z]`
- 沒有 `visibility`
- 若遠端只保存 `skeleton_sequence`，最新單幀可由 `skeleton_sequence.frames[-1]` 取得

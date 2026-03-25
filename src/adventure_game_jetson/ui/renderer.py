from __future__ import annotations

import os
import subprocess
import textwrap
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFont

from adventure_game_jetson.core.types import ActionPrediction, GameSnapshot


class GameRenderer:
    def __init__(
        self,
        window_name: str = "Adventure Game Jetson",
        hp_max: int = 10,
        font_path: str = "",
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self.window_name = window_name
        self.hp_max = int(hp_max)
        self.font_path = font_path
        self.win_w = int(width)
        self.win_h = int(height)
        self._hp_anim = float(self.hp_max)
        self._hp_ghost = float(self.hp_max)
        self._prev_hp = self.hp_max
        self._hit_t = -1.0
        self._font_cache: dict[int, Any] = {}
        self._text_sprite_cache: dict[tuple[str, int, tuple[int, int, int]], tuple[np.ndarray, np.ndarray, int, int]] = {}
        self._init_fonts()
        window_flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        cv2.namedWindow(self.window_name, window_flags)
        cv2.resizeWindow(self.window_name, self.win_w, self.win_h)

    def _init_fonts(self) -> None:
        if self.font_path and os.path.exists(self.font_path):
            return

        preferred_queries = [
            "Noto Sans CJK TC",
            "Noto Sans CJK SC",
            "Noto Serif CJK TC",
        ]
        for query in preferred_queries:
            try:
                result = subprocess.run(
                    ["fc-match", query, "-f", "%{file}\n"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                path = result.stdout.strip()
            except Exception:
                path = ""
            if path and os.path.exists(path):
                self.font_path = path
                return

        potential_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ]
        for path in potential_paths:
            if path and os.path.exists(path):
                self.font_path = path
                return
        self.font_path = ""

    def _get_font(self, size: int):
        cached = self._font_cache.get(size)
        if cached is not None:
            return cached
        try:
            font = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def _get_text_sprite(
        self,
        text: str,
        size: int,
        color: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        key = (text, size, color)
        cached = self._text_sprite_cache.get(key)
        if cached is not None:
            return cached

        font = self._get_font(size)
        left, top, right, bottom = font.getbbox(text)
        pad = max(8, size // 5)
        width = max(1, right - left) + pad * 2
        height = max(1, bottom - top) + pad * 2
        image = PilImage.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text(
            (-left + pad, -top + pad),
            text,
            font=font,
            fill=(color[2], color[1], color[0], 255),
        )
        rgba = np.array(image, dtype=np.uint8)
        bgr = rgba[:, :, :3][:, :, ::-1].copy()
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        sprite = (bgr, alpha, left - pad, top - pad)
        self._text_sprite_cache[key] = sprite
        return sprite

    def _draw_text_cn(
        self,
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        size: int = 20,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        if not text:
            return img
        sprite_bgr, alpha, left, top = self._get_text_sprite(text, size, color)
        x0 = max(0, pos[0] + left)
        y0 = max(0, pos[1] + top)
        x1 = min(img.shape[1], x0 + sprite_bgr.shape[1])
        y1 = min(img.shape[0], y0 + sprite_bgr.shape[0])
        if x0 >= x1 or y0 >= y1:
            return img

        sprite_w = x1 - x0
        sprite_h = y1 - y0
        roi = img[y0:y1, x0:x1]
        sprite_roi = sprite_bgr[:sprite_h, :sprite_w]
        alpha_roi = alpha[:sprite_h, :sprite_w, None]
        blended = roi.astype(np.float32) * (1.0 - alpha_roi) + sprite_roi.astype(np.float32) * alpha_roi
        img[y0:y1, x0:x1] = blended.astype(np.uint8)
        return img

    def _draw_stickman(self, img: np.ndarray, skeleton: np.ndarray | None, cam_w: int) -> None:
        if skeleton is None or len(skeleton) == 0:
            return
        connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
        points: list[tuple[int, int] | None] = []
        for x, y, _z in skeleton:
            if x == 0.0 and y == 0.0:
                points.append(None)
                continue
            points.append((int(x * cam_w), int(y * self.win_h)))
        for start, end in connections:
            if start < len(points) and end < len(points) and points[start] and points[end]:
                cv2.line(img, points[start], points[end], (0, 255, 0), 3)
        for point in points:
            if point:
                cv2.circle(img, point, 5, (0, 255, 0), -1)

    def _draw_hp_segments(
        self,
        hud: np.ndarray,
        x: int,
        y: int,
        seg_w: int,
        seg_h: int,
        hp_anim: float,
        gap: int,
        low_flash: bool,
    ) -> None:
        if self._hp_ghost > hp_anim:
            self._hp_ghost -= 0.05
        else:
            self._hp_ghost = hp_anim
        for index in range(self.hp_max):
            base_x = x + index * (seg_w + gap)
            cv2.rectangle(hud, (base_x, y), (base_x + seg_w, y + seg_h), (40, 40, 40), -1)
            if index < int(self._hp_ghost):
                cv2.rectangle(hud, (base_x, y), (base_x + seg_w, y + seg_h), (0, 0, 150), -1)
            elif index == int(self._hp_ghost):
                partial = int(seg_w * (self._hp_ghost - index))
                cv2.rectangle(hud, (base_x, y), (base_x + partial, y + seg_h), (0, 0, 150), -1)
            color = (0, 0, 255) if low_flash else (0, 255, 0)
            if index < int(hp_anim):
                cv2.rectangle(hud, (base_x, y), (base_x + seg_w, y + seg_h), color, -1)
            elif index == int(hp_anim):
                partial = int(seg_w * (hp_anim - index))
                cv2.rectangle(hud, (base_x, y), (base_x + partial, y + seg_h), color, -1)
            cv2.rectangle(hud, (base_x, y), (base_x + seg_w, y + seg_h), (90, 90, 90), 2)

    def _apply_shake_impact(self, img: np.ndarray, now: float, hit_age: float) -> np.ndarray:
        max_duration = 0.4
        if hit_age > max_duration:
            return img
        decay = (1.0 - hit_age / max_duration) ** 2
        intensity = 25 * decay
        dx = int(np.random.uniform(-1, 1) * intensity)
        dy = int(np.random.uniform(-1, 1) * intensity)
        if hit_age < 0.1:
            overlay = img.copy()
            overlay[:] = (0, 0, 150)
            img = cv2.addWeighted(overlay, 0.3 * decay, img, 1.0, 0)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    def render(
        self,
        frame: np.ndarray | None,
        skeleton: np.ndarray | None,
        snapshot: GameSnapshot,
        action_prediction: ActionPrediction | None,
    ) -> np.ndarray:
        scale = max(0.5, min(self.win_w / 1920.0, self.win_h / 1080.0))

        def px(value: float, minimum: int = 1) -> int:
            return max(minimum, int(round(value * scale)))

        now = time.time()
        hp = snapshot.hp
        if hp < self._prev_hp:
            self._hit_t = now
        self._prev_hp = hp
        self._hp_anim += (hp - self._hp_anim) * 0.1

        ui = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        hud_w = int(self.win_w * 0.40)
        cam_w = self.win_w - hud_w

        if frame is not None:
            cam_view = cv2.resize(frame, (cam_w, self.win_h))
            self._draw_stickman(cam_view, skeleton, cam_w)
            ui[:, :cam_w] = cam_view
        else:
            cv2.putText(ui, "WAITING FOR CAMERA...", (100, self.win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.line(ui, (cam_w, 0), (cam_w, self.win_h), (60, 60, 60), 3)

        hud = ui[:, cam_w:]
        hud[:] = (25, 25, 30)
        hud = self._draw_text_cn(hud, "AI 體感冒險王", (px(30), px(35)), size=px(36), color=(255, 255, 0))
        cv2.putText(hud, f"STATE: {snapshot.state}", (px(30), px(95)), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, 0.7 * scale), (200, 200, 200), max(1, px(2)))
        cv2.putText(hud, f"SCORE: {snapshot.score}", (px(300), px(95)), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, 0.7 * scale), (0, 255, 255), max(1, px(2)))

        low_flash = hp <= 1 and int(now * 5) % 2 == 0
        available_w = hud_w - px(60)
        gap = px(8)
        seg_w = int((available_w - (gap * max(0, self.hp_max - 1))) / max(1, self.hp_max))
        seg_w = min(px(70), max(px(10), seg_w))
        self._draw_hp_segments(hud, px(30), px(115), seg_w, px(20), self._hp_anim, gap, low_flash)

        overlay = hud.copy()
        cv2.rectangle(overlay, (px(20), px(160)), (hud_w - px(20), px(600)), (45, 45, 55), -1)
        hud = cv2.addWeighted(overlay, 0.6, hud, 0.4, 0)
        cv2.rectangle(hud, (px(20), px(160)), (hud_w - px(20), px(600)), (80, 80, 80), max(1, px(1)))
        hud = self._draw_text_cn(hud, "劇情 Story:", (px(30), px(175)), size=px(22), color=(150, 200, 255))

        wrapped_lines = textwrap.wrap(snapshot.narration, width=28)
        y_text = px(215)
        for line in wrapped_lines:
            if y_text > px(580):
                break
            hud = self._draw_text_cn(hud, line, (px(30), y_text), size=px(24), color=(230, 230, 230))
            y_text += px(32)

        panel_y = px(620)
        panel_h = px(180)
        if snapshot.state in {"PREPARING", "EVENT"}:
            border_color = (0, 0, 255) if (snapshot.state == "EVENT" and snapshot.time_left_sec < 3.0 and int(now * 10) % 2 == 0) else (0, 255, 255)
            cv2.rectangle(hud, (px(20), panel_y), (hud_w - px(20), panel_y + panel_h), (30, 30, 30), -1)
            cv2.rectangle(hud, (px(20), panel_y), (hud_w - px(20), panel_y + panel_h), border_color, max(1, px(4)))
            hud = self._draw_text_cn(hud, "請做動作 Action:", (px(40), panel_y + px(40)), size=px(22), color=(200, 200, 200))
            cv2.putText(
                hud,
                snapshot.active_action.upper(),
                (px(40), panel_y + px(100)),
                cv2.FONT_HERSHEY_TRIPLEX,
                max(0.9, 2.0 * scale),
                (0, 255, 255),
                max(1, px(3)),
            )
            if snapshot.state == "PREPARING":
                hud = self._draw_text_cn(hud, "準備中 Get Ready...", (px(40), panel_y + px(150)), size=px(24), color=(255, 255, 0))
                if int(now * 6) % 2 == 0:
                    ui = self._draw_text_cn(ui, "準備!", (cam_w // 2 - px(150), self.win_h // 2), size=px(120), color=(0, 255, 255))
            else:
                cv2.putText(
                    hud,
                    f"TIME: {snapshot.time_left_sec:.1f}s",
                    (px(40), panel_y + px(150)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.65, 1.0 * scale),
                    (255, 255, 255),
                    max(1, px(2)),
                )
        else:
            cv2.rectangle(hud, (px(20), panel_y), (hud_w - px(20), panel_y + panel_h), (20, 20, 20), -1)
            cv2.rectangle(hud, (px(20), panel_y), (hud_w - px(20), panel_y + panel_h), (50, 50, 50), max(1, px(1)))
            if snapshot.state == "STORY":
                hud = self._draw_text_cn(hud, "請閱讀劇情...", (px(40), panel_y + px(100)), size=px(26), color=(100, 100, 100))
            elif snapshot.state == "RESULT":
                result_color = (0, 255, 0) if snapshot.result_success else (0, 0, 255)
                hud = self._draw_text_cn(hud, "結算中...", (px(40), panel_y + px(100)), size=px(30), color=result_color)

        y_base = px(850)
        cv2.putText(hud, "SYSTEM SEES:", (px(30), y_base), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.6 * scale), (100, 100, 100), max(1, px(1)))
        if action_prediction:
            is_correct = action_prediction.action == snapshot.active_action and snapshot.state == "EVENT"
            color = (0, 255, 0) if is_correct else (180, 180, 180)
            cv2.putText(
                hud,
                action_prediction.action.upper(),
                (px(30), y_base + px(50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.8, 1.2 * scale),
                color,
                max(1, px(2)),
            )
            cv2.putText(
                hud,
                f"CONF: {action_prediction.confidence:.2f}",
                (px(30), y_base + px(90)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.6 * scale),
                (150, 150, 150),
                max(1, px(1)),
            )

        ui[:, cam_w:] = hud
        if self._hit_t > 0:
            ui = self._apply_shake_impact(ui, now, now - self._hit_t)
        if snapshot.game_over:
            overlay = ui.copy()
            cv2.rectangle(overlay, (0, self.win_h // 2 - px(150)), (self.win_w, self.win_h // 2 + px(150)), (0, 0, 0), -1)
            ui = cv2.addWeighted(overlay, 0.75, ui, 0.25, 0)
            ui = self._draw_text_cn(ui, "遊戲結束 GAME OVER", (self.win_w // 2 - px(300), self.win_h // 2 - px(50)), size=px(80), color=(0, 0, 255))
        return ui

    def show(self, image: np.ndarray) -> int:
        cv2.imshow(self.window_name, image)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()

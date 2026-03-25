from __future__ import annotations

import random
import time

from .story import EVENTS_DB, NO_EVENT_AFTER, StoryTeller
from .types import ActionPrediction, GameEvent, GameSnapshot


class GameEngine:
    def __init__(
        self,
        min_conf: float = 0.6,
        hp_init: int = 10,
        story_duration: float = 5.0,
        prep_duration: float = 2.0,
        event_duration: float = 15.0,
        result_duration: float = 3.0,
        run_duration: float = 15.0,
        ending_duration: float = 15.0,
        openai_api_key: str = "",
    ) -> None:
        self.min_conf = float(min_conf)
        self.hp_init = int(hp_init)
        self.t_story = float(story_duration)
        self.t_prep = float(prep_duration)
        self.t_event = float(event_duration)
        self.t_result = float(result_duration)
        self.t_run = float(run_duration)
        self.t_ending = float(ending_duration)
        self.story_teller = StoryTeller(openai_api_key)
        self.max_loop = 27
        self.reset()

    def reset(self) -> None:
        self.hp = self.hp_init
        self.score = 0
        self.current_loop = 1
        self.state = "IDLE"
        self.active_event: dict[str, object] | None = None
        self.deadline = 0.0
        self.next_state_time = 0.0
        self.curr_evt_start_t = 0.0
        self.last_random_req: str | None = None
        self.is_extra_run = False
        self.last_action = ActionPrediction()
        self.narration = "系統啟動中... 等待劇情載入..."
        self.last_result_success: bool | None = None

    def submit_action(self, action: ActionPrediction | None) -> None:
        if action is not None:
            self.last_action = action

    def tick(self, now: float | None = None) -> GameSnapshot:
        now = time.monotonic() if now is None else float(now)
        time_left = max(0.0, self.deadline - now) if self.deadline else 0.0

        if self.hp <= 0:
            if self.state != "GAME_OVER":
                self.state = "GAME_OVER"
                self.narration = "GAME OVER"
            return self.snapshot(time_left)

        if self.state == "GAME_OVER":
            return self.snapshot(time_left)

        if self.state == "IDLE":
            self.current_loop = 1
            self.enter_story(now)
        elif self.state == "STORY":
            if now >= self.next_state_time:
                self.check_next_step(now)
        elif self.state == "PREPARING":
            if now >= self.next_state_time:
                self.enter_event(now)
        elif self.state == "EVENT":
            is_fresh = self.last_action.produced_at > self.curr_evt_start_t
            if self.current_loop == 27:
                if (
                    self.last_action.action == "run_forward"
                    and self.last_action.confidence >= self.min_conf
                    and is_fresh
                ):
                    self.resolve_end(True, now)
                elif now >= self.deadline:
                    self.resolve_end(False, now)
            elif self.active_event:
                if (
                    self.last_action.action == self.active_event["req"]
                    and self.last_action.confidence >= self.min_conf
                    and is_fresh
                ):
                    self.resolve_event(True, now)
                elif now >= self.deadline:
                    self.resolve_event(False, now)
        elif self.state == "RESULT":
            if now >= self.next_state_time:
                if (
                    self.current_loop >= 13
                    and self.current_loop not in NO_EVENT_AFTER
                    and not self.is_extra_run
                ):
                    self.trigger_run_event(now)
                else:
                    self.is_extra_run = False
                    self.advance_loop(now)
        elif self.state == "ENDING":
            if now >= self.next_state_time:
                self.state = "GAME_OVER"
                self.narration = "GAME OVER"

        if self.state == "EVENT":
            time_left = max(0.0, self.deadline - now)
        elif self.state == "PREPARING":
            time_left = max(0.0, self.next_state_time - now)
        else:
            time_left = 0.0
        return self.snapshot(time_left)

    def snapshot(self, time_left: float = 0.0) -> GameSnapshot:
        active_action = ""
        event_prompt = ""
        if self.active_event and self.state in {"PREPARING", "EVENT"}:
            active_action = str(self.active_event.get("req", ""))
            event_prompt = str(self.active_event.get("text", ""))
        return GameSnapshot(
            state=self.state,
            hp=self.hp,
            score=self.score,
            current_loop=self.current_loop,
            time_left_sec=float(time_left),
            active_action=active_action,
            narration=self.narration,
            event_prompt=event_prompt,
            result_success=self.last_result_success,
            game_over=self.state == "GAME_OVER",
        )

    def current_event(self) -> GameEvent:
        if not self.active_event:
            return GameEvent()
        return GameEvent(
            prompt=str(self.active_event.get("text", "")),
            required_action=str(self.active_event.get("req", "")),
            time_limit_sec=float(self.active_event.get("time", 0.0)),
        )

    def enter_story(self, now: float) -> None:
        self.state = "STORY"
        self.active_event = None
        self.last_result_success = None
        self.narration = f"[Loop {self.current_loop}] {self.story_teller.generate_text(self.current_loop)}"
        self.next_state_time = now + self.t_story

    def check_next_step(self, now: float) -> None:
        if self.current_loop in NO_EVENT_AFTER and self.current_loop != 27:
            self.advance_loop(now)
            return

        self.state = "PREPARING"
        self.next_state_time = now + self.t_prep
        self.is_extra_run = False
        self.last_result_success = None

        if self.current_loop == 11:
            self.active_event = {
                "text": "巨石滾來！快跑！",
                "req": "run_forward",
                "time": 10.0,
                "s_text": "你驚險地避開了巨石!",
                "f_text": "巨石擦撞到了你!",
            }
        elif self.current_loop == 13:
            self.active_event = {
                "text": "拿寶劍的人追來了！快跑！",
                "req": "run_forward",
                "time": 10.0,
                "s_text": "你拉開了距離!",
                "f_text": "你被劍氣傷到了!",
            }
        elif self.current_loop == 27:
            self.active_event = {"text": "選擇: 跑(離開) / 不動(拿劍)", "req": "run_forward", "time": 25.0}
        else:
            chapter = 1 if self.current_loop <= 5 else (2 if self.current_loop <= 12 else 3)
            pool = [item for item in EVENTS_DB if item["chapter"] == chapter]
            valid_pool = [item for item in pool if item["req"] != self.last_random_req] or pool
            self.active_event = dict(random.choice(valid_pool))
            self.active_event["time"] = self.t_event
            self.last_random_req = str(self.active_event["req"])

        if self.active_event:
            self.narration = f"【突發事件】\n{self.active_event['text']}"

    def trigger_run_event(self, now: float) -> None:
        self.state = "PREPARING"
        self.is_extra_run = True
        self.next_state_time = now + 2.0
        self.last_result_success = None
        self.active_event = {
            "text": "還沒結束！快跑！",
            "req": "run_forward",
            "time": self.t_run,
            "s_text": "呼...好險",
            "f_text": "太慢了!",
        }
        self.narration = f"【追逐戰】\n{self.active_event['text']}"

    def enter_event(self, now: float) -> None:
        if not self.active_event:
            return
        self.state = "EVENT"
        self.deadline = now + float(self.active_event["time"])
        self.curr_evt_start_t = now

    def resolve_event(self, success: bool, now: float) -> None:
        if not self.active_event:
            return
        self.state = "RESULT"
        self.last_result_success = success
        if success:
            self.score += 10
            self.narration = str(self.active_event.get("s_text", "Success!"))
        else:
            self.hp -= 1
            self.narration = str(self.active_event.get("f_text", "Fail!"))
        self.next_state_time = now + self.t_result

    def resolve_end(self, is_run: bool, now: float) -> None:
        self.last_result_success = is_run
        self.narration = self.story_teller.generate_end(1 if is_run else 2)
        self.state = "ENDING"
        self.next_state_time = now + self.t_ending

    def advance_loop(self, now: float) -> None:
        self.current_loop += 1
        if self.current_loop > self.max_loop:
            self.state = "GAME_OVER"
            self.narration = "GAME OVER"
        else:
            self.enter_story(now)

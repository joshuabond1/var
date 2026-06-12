"""
game.py – Rules engine and game state for Garden Referee

Custom rules (fully autonomous):
  1. FIRST-TIME FINISHING ONLY:
       Detected automatically via ball-speed profiling.
       If the ball's speed drops below the "controlled" threshold while
       near a player, that player is flagged as having controlled it.
       A subsequent shot from that player in the same phase → FOUL.

  2. THREE-PASS MINIMUM (dead ball & keeper pickup):
       Dead ball:  ball leaves the pitch polygon for 5+ consecutive frames.
       Keeper pickup: ball is stationary inside/near a goal zone for 6+ frames.
       Both reset the pass counter to 0 and require 3 passes before a goal counts.
"""

import time
import json
import math
import random
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
POSSESSION_RADIUS_PX    = 90    # px — ball within this of a player = possessed
POSSESSION_DEBOUNCE_S   = 0.35  # min seconds between possession changes
BALL_SPEED_CONTROLLED   = 5.0   # px/frame — ball this slow near player = controlled
BALL_SPEED_SHOT         = 16.0  # px/frame — ball this fast = possible shot
KEEPER_ZONE_BUFFER_PX   = 80    # px beyond goal polygon = keeper zone
KEEPER_STILL_FRAMES     = 6     # frames ball must be still in keeper zone
OOB_FRAMES_THRESHOLD    = 5     # consecutive frames out-of-bounds → dead ball


# ---------------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------------
_COMMENTARY = {
    "goal":            ["GOAL! AI Referee confirms it! 🎉",
                        "The net bulges! GOAL! 🚀",
                        "GET IN! Referee confirms: GOAL! ⚽"],
    "foul_first_time": ["FOUL! Ball was controlled — first-time finishing only! 🚫",
                        "Nope! You trapped it first. First-touch only in this garden! 🚫",
                        "REF BLOWS WHISTLE — controlled ball, not first-time! 🟥"],
    "foul_3pass":      ["FOUL! Only {n}/3 passes from dead ball — keep it moving!",
                        "Not enough passes! {n}/3 — need 3 before you shoot.",
                        "Hold on — {n} pass(es) counted. Need at least 3!"],
    "pass_dead":       ["Pass {n}/3 — keep going!",
                        "Good ball! {n}/3 done.",
                        "Ball moving! {n}/3 passes completed."],
    "pass_live":       ["Good combination play.",
                        "Neat touch.",
                        "Quick interchange."],
    "keeper_auto":     ["🧤 Keeper has it — AI detects ball held in goal zone. Pass counter reset.",
                        "🧤 Ball stationary in goal area — keeper pickup detected. 0/3 required.",
                        "🧤 Monkey rush keeper collects — 3-pass rule resets to 0/3!"],
    "dead_ball_auto":  ["🚩 Ball out of play — AI detects out-of-bounds. 3-pass rule resets.",
                        "🚩 Out! Ball left the pitch. Dead ball — 3 passes required.",
                        "🚩 Off the pitch — dead ball detected automatically."],
    "var":             ["Checking the VAR…",
                        "Going upstairs for a look…",
                        "VAR in progress — hold that celebration!"],
    "kickoff":         ["🟢 Garden FC AI Referee is LIVE. Let the chaos begin! ⚽",
                        "🟢 Welcome to the garden. The ref is watching every touch. 👁️"],
}


def _say(key: str, **kwargs) -> str:
    lines = _COMMENTARY.get(key, [key])
    return random.choice(lines).format(**kwargs)


# ---------------------------------------------------------------------------
# Goal descriptor
# ---------------------------------------------------------------------------
@dataclass
class Goal:
    id: int
    poly: list   # [[x,y], ...] 4 points
    label: str

    def center(self):
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        return (sum(xs) // len(xs), sum(ys) // len(ys))

    def ball_inside(self, pos: tuple) -> bool:
        return self._test(pos, 0)

    def ball_near(self, pos: tuple, buf: int = KEEPER_ZONE_BUFFER_PX) -> bool:
        return self._test(pos, buf)

    def _test(self, pos: tuple, buf: int) -> bool:
        import cv2
        if not self.poly:
            return False
        pts = np.array(self.poly, dtype=np.float32)
        bx, by = float(pos[0]), float(pos[1])
        dist = cv2.pointPolygonTest(pts, (bx, by), True)
        return dist >= -buf


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, config_path: str = "pitch_config.json"):
        # Scores — keyed by team
        self.score = {"home": 0, "away": 0}

        # Team assignment: player track_id → "home" | "away" | None
        self.team_of: dict[int, Optional[str]] = {}

        # Phase
        self.phase = "dead_ball"

        # Pass tracking
        self.pass_count      = 0
        self.required_passes = 3
        self.possession_chain: list[int] = []
        self.current_possessor: Optional[int] = None
        self._last_pos_change = 0.0

        # Per-player controlled-ball flag (reset each dead ball)
        self._player_controlled: dict[int, bool] = {}

        # Auto dead-ball counters
        self._oob_frames      = 0   # consecutive out-of-bounds frames
        self._keeper_frames   = 0   # consecutive frames ball still in keeper zone

        # Ball history (speed + trajectory for goal-line tech)
        self._ball_speed_hist: deque = deque(maxlen=12)
        self._ball_trajectory: deque = deque(maxlen=4)  # recent ball positions

        # UI / events
        self.commentary: deque = deque(maxlen=50)
        self.last_foul:  Optional[str] = None
        self.last_goal:  Optional[int] = None
        self.var_active  = False

        # Pitch polygon (set from server after loading config)
        self.pitch_polygon: Optional[np.ndarray] = None

        # Goal-line tech cooldown (instance-level, not class-level)
        self._goal_cooldown: dict[int, float] = {}

        self.goals: list[Goal] = []
        self._load_config(config_path)
        self._add_raw(_say("kickoff"))

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_config(self, path: str):
        try:
            with open(path) as f:
                cfg = json.load(f)
            for g in cfg.get("goals", []):
                self.goals.append(Goal(
                    id=g["id"],
                    poly=g["poly"],
                    label=g.get("label", f"Goal {g['id']}"),
                ))
            pts = cfg.get("pitch_polygon", [])
            if pts:
                self.pitch_polygon = np.array(pts, dtype=np.int32)
            print(f"[Game] Config loaded: {len(self.goals)} goal(s), "
                  f"pitch polygon: {'yes' if self.pitch_polygon is not None else 'no'}")
        except FileNotFoundError:
            print("[Game] No pitch_config.json — run setup_pitch.py first!")
        except Exception as e:
            print(f"[Game] Config error: {e}")

    # ------------------------------------------------------------------
    # Team assignment (called by server from shirt-colour clustering)
    # ------------------------------------------------------------------
    def assign_team(self, player_id: int, team: str):
        self.team_of[player_id] = team

    def team_label(self, player_id: int) -> str:
        return self.team_of.get(player_id, "?")

    # ------------------------------------------------------------------
    # Main update — called every processed frame
    # ------------------------------------------------------------------
    def update(
        self,
        players: list,
        ball,
        ball_speed: float,
    ) -> Optional[str]:
        self._ball_speed_hist.append(ball_speed)
        if ball is not None:
            self._ball_trajectory.append(ball.center)

        # ---- Auto: out-of-bounds dead ball ---------------------------
        oob_event = self._check_out_of_bounds(ball)
        if oob_event:
            return oob_event

        if ball is None or not players:
            return None

        # ---- Auto: keeper pickup ------------------------------------
        keeper_event = self._check_keeper_pickup(ball, ball_speed)
        if keeper_event:
            return keeper_event

        # ---- Auto: goal-line technology --------------------------------
        goal_event = self._check_goal(ball)
        if goal_event:
            return goal_event

        # ---- Possession + passes ------------------------------------
        return self._track_possession(players, ball, ball_speed)

    # ------------------------------------------------------------------
    # Auto: out-of-bounds dead ball
    # ------------------------------------------------------------------
    def _check_out_of_bounds(self, ball) -> Optional[str]:
        if ball is None or self.pitch_polygon is None:
            self._oob_frames = 0
            return None

        import cv2
        bx, by = float(ball.center[0]), float(ball.center[1])
        inside = cv2.pointPolygonTest(self.pitch_polygon, (bx, by), False) >= 0

        if not inside:
            self._oob_frames += 1
            if self._oob_frames == OOB_FRAMES_THRESHOLD:
                return self._auto_dead_ball()
        else:
            self._oob_frames = 0
        return None

    def _auto_dead_ball(self) -> str:
        self.phase = "dead_ball"
        self.pass_count = 0
        self.possession_chain = []
        self.current_possessor = None
        self._player_controlled = {}
        self._oob_frames = 0
        self._add_raw(_say("dead_ball_auto"))
        return "dead_ball"

    # ------------------------------------------------------------------
    # Auto: keeper pickup
    # ------------------------------------------------------------------
    def _check_keeper_pickup(self, ball, ball_speed: float) -> Optional[str]:
        if not self.goals:
            return None

        near_goal = any(g.ball_near(ball.center) for g in self.goals)
        if near_goal and ball_speed < BALL_SPEED_CONTROLLED:
            self._keeper_frames += 1
            if self._keeper_frames == KEEPER_STILL_FRAMES:
                return self._auto_keeper_pickup()
        else:
            self._keeper_frames = 0
        return None

    def _auto_keeper_pickup(self) -> str:
        self.phase = "dead_ball"
        self.pass_count = 0
        self.possession_chain = []
        self.current_possessor = None
        self._player_controlled = {}
        self._keeper_frames = 0
        self._add_raw(_say("keeper_auto"))
        return "keeper_pickup"

    # ------------------------------------------------------------------
    # Goal-line technology
    # ------------------------------------------------------------------
    _GOAL_COOLDOWN_S = 4.0

    def _check_goal(self, ball) -> Optional[str]:
        """
        Two-layer goal detection:
          1. Trajectory crossing: does the ball's path this frame cross the
             goal line segment (between the two posts)?  Catches fast shots.
          2. Ball-inside-polygon: ball centre is inside the goal polygon.
             Guards against the trajectory check missing a slow ball.

        Both require the ball to have sufficient speed to filter out keeper
        holding the ball in front of goal.
        """
        avg_speed = (sum(self._ball_speed_hist) / len(self._ball_speed_hist)
                     if self._ball_speed_hist else 0)
        if avg_speed < 2.5:
            return None

        now = time.time()

        for goal in self.goals:
            gid = goal.id
            last = self._goal_cooldown.get(gid, 0)
            if (now - last) < self._GOAL_COOLDOWN_S:
                continue

            crossed = False

            # Layer 1: trajectory crossing (line-segment intersection)
            if len(self._ball_trajectory) >= 2:
                prev_pos = self._ball_trajectory[-2]
                curr_pos = self._ball_trajectory[-1]
                if self._segment_crosses_goal_line(prev_pos, curr_pos, goal):
                    crossed = True

            # Layer 2: ball centre inside goal polygon
            if not crossed and goal.ball_inside(ball.center):
                crossed = True

            if crossed:
                self._goal_cooldown[gid] = now
                return self._confirm_goal_auto(gid)

        return None

    @staticmethod
    def _segment_crosses_goal_line(p1: tuple, p2: tuple, goal: 'Goal') -> bool:
        """
        Returns True if the segment p1→p2 crosses the front face of the goal
        (the line between poly[0] and poly[1], i.e. the two post positions).
        Uses 2D line-segment intersection.
        """
        if len(goal.poly) < 2:
            return False

        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersects(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        g0 = tuple(goal.poly[0])
        g1 = tuple(goal.poly[1])
        return intersects(p1, p2, g0, g1)

    def _confirm_goal_auto(self, goal_id: int) -> str:
        # Goal against whichever team's goal was scored in
        # Goal 0 = team "away" scored (home goal) → home concedes → away scores
        # Goal 1 = team "home" scored (away goal) → away concedes → home scores
        # Convention: goal_id 0 → home team's goal (so away scores)
        #             goal_id 1 → away team's goal (so home scores)
        scorer = "away" if goal_id == 0 else "home"
        self.score[scorer] += 1
        self.last_goal = goal_id
        self.phase = "dead_ball"
        self.pass_count = 0
        self.possession_chain = []
        self.current_possessor = None
        self._player_controlled = {}
        self._add_raw(_say("goal"))
        return f"goal_{goal_id}"

    # ------------------------------------------------------------------
    # Possession tracking + first-time rule
    # ------------------------------------------------------------------
    def _track_possession(self, players, ball, ball_speed: float) -> Optional[str]:
        bx, by = ball.center
        now = time.time()

        closest, closest_dist = None, float("inf")
        for p in players:
            d = math.hypot(p.center[0] - bx, p.center[1] - by)
            if d < closest_dist:
                closest_dist = d
                closest = p

        if closest is None or closest_dist > POSSESSION_RADIUS_PX:
            return None

        tid = closest.track_id

        if tid == self.current_possessor:
            # Track whether this player has slowed/controlled the ball
            if ball_speed < BALL_SPEED_CONTROLLED:
                self._player_controlled[tid] = True
            return None

        # ---- Possession change ----------------------------------------
        if (now - self._last_pos_change) < POSSESSION_DEBOUNCE_S:
            return None

        prev = self.current_possessor
        self.current_possessor = tid
        self._last_pos_change = now

        # Initialise control state for incoming player
        if tid not in self._player_controlled:
            self._player_controlled[tid] = ball_speed < BALL_SPEED_CONTROLLED

        if prev is None:
            self.possession_chain = [tid]
            return None

        # It's a pass
        if tid not in self.possession_chain:
            self.possession_chain.append(tid)

        if prev != tid:
            self.pass_count += 1
            if self.phase == "dead_ball" and self.pass_count >= self.required_passes:
                self.phase = "live"
            if self.phase == "dead_ball":
                self._add_raw(_say("pass_dead", n=self.pass_count))
            else:
                self._add_raw(_say("pass_live"))
            return "pass"

        return None

    # ------------------------------------------------------------------
    # Shot validation — called from server when shot detected
    # ------------------------------------------------------------------
    def validate_shot(self, shooter_id: int, toward_goal_id: int) -> dict:
        # Rule 1: First-time finishing
        was_controlled = self._player_controlled.get(shooter_id, False)
        if was_controlled:
            self.last_foul = "first_time"
            msg = _say("foul_first_time")
            self._add_raw(msg)
            self.phase = "dead_ball"
            self.pass_count = 0
            self.possession_chain = []
            self._player_controlled = {}
            return {"valid": False, "reason": "foul_first_time", "message": msg}

        # Rule 2: 3-pass minimum from dead ball
        if self.phase == "dead_ball":
            msg = _say("foul_3pass", n=self.pass_count)
            self.last_foul = "3pass"
            self._add_raw(msg)
            return {"valid": False, "reason": "foul_3pass", "message": msg}

        return {"valid": True, "reason": "ok", "message": "Shot valid!"}

    # ------------------------------------------------------------------
    # Manual overrides (still available as referee buttons)
    # ------------------------------------------------------------------
    def manual_keeper_pickup(self) -> dict:
        msg = "🧤 Manual: Keeper pickup — pass counter reset."
        self.phase = "dead_ball"
        self.pass_count = 0
        self.possession_chain = []
        self._player_controlled = {}
        self._add_raw(msg)
        return {"message": msg}

    def manual_dead_ball(self) -> dict:
        msg = "🚩 Manual: Dead ball — 3 passes required."
        self.phase = "dead_ball"
        self.pass_count = 0
        self.possession_chain = []
        self._player_controlled = {}
        self._add_raw(msg)
        return {"message": msg}

    def manual_goal(self, goal_id: int) -> dict:
        result_str = self._confirm_goal_auto(goal_id)
        return {"score": self.score, "message": self.commentary[-1]["text"]}

    def trigger_var(self) -> dict:
        self.var_active = True
        msg = _say("var")
        self._add_raw(msg)
        return {"message": msg}

    def cancel_var(self) -> dict:
        self.var_active = False
        return {"message": "VAR check complete."}

    # ------------------------------------------------------------------
    # State snapshot for UI
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            "score":             self.score,
            "phase":             self.phase,
            "pass_count":        self.pass_count,
            "required_passes":   self.required_passes,
            "current_possessor": self.current_possessor,
            "possessor_team":    self.team_of.get(self.current_possessor),
            "team_of":           {str(k): v for k, v in self.team_of.items()},
            "last_foul":         self.last_foul,
            "last_goal":         self.last_goal,
            "var_active":        self.var_active,
            "commentary":        list(self.commentary)[-10:],
        }

    # ------------------------------------------------------------------
    # Commentary
    # ------------------------------------------------------------------
    def _add_raw(self, text: str):
        self.commentary.append({
            "time": time.strftime("%H:%M:%S"),
            "text": text,
        })

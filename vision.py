"""
vision.py – CV pipeline for Garden Referee

Player detection + multi-person pose overlay: YOLOv8n-pose
Ball detection: YOLO sports-ball class → HSV colour fallback
Shirt colour team detection: torso-pixel HSV sampling + 2-cluster assignment
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional
from collections import deque, defaultdict

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerDetection:
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2)
    center: tuple        # (cx, cy)
    keypoints: Optional[np.ndarray] = None  # (17, 3): x, y, conf
    shirt_hue: Optional[float] = None       # median HSV hue 0-179


@dataclass
class BallDetection:
    center: tuple
    radius: float
    confidence: float
    source: str          # "yolo" | "hsv"


# ---------------------------------------------------------------------------
# Skeleton config
# ---------------------------------------------------------------------------
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

SPORTS_BALL_CLASS = 32

BALL_HSV_RANGES = [
    {"lower": np.array([0,   0, 200]), "upper": np.array([180, 40, 255])},  # white
    {"lower": np.array([5, 100, 100]), "upper": np.array([25, 255, 255])},  # orange
]

# Team colours used for bounding-box tint
TEAM_COLOURS = {
    "home": (255, 80,  80),   # red-ish
    "away": (80,  160, 255),  # blue-ish
    None:   (200, 200, 200),  # unknown
}


# ---------------------------------------------------------------------------
# Shirt-colour team detector
# ---------------------------------------------------------------------------

class ShirtColourDetector:
    """
    Every frame, sample the torso region of each tracked player to get their
    shirt's dominant HSV hue. After collecting enough samples, split players
    into 2 teams (home/away) by finding the biggest hue gap.

    Re-clusters every RECLUSTER_FRAMES frames to handle new players.
    """
    SAMPLE_FRAMES   = 45    # frames to collect before first cluster
    RECLUSTER_FRAMES = 180  # re-cluster every N frames

    def __init__(self):
        # player_id → deque of median hue values
        self._hue_samples: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self._frame_count = 0
        self._last_cluster: dict[int, str] = {}   # player_id → "home"|"away"

    def update(self, frame: np.ndarray, players: list[PlayerDetection]) -> dict[int, str]:
        """
        Update hue samples for all players and return current team assignments.
        """
        self._frame_count += 1

        for p in players:
            hue = self._sample_torso(frame, p)
            if hue is not None:
                self._hue_samples[p.track_id].append(hue)
                p.shirt_hue = hue

        # Cluster when we have enough data, or at recluster interval
        enough = all(len(v) >= 5 for v in self._hue_samples.values()
                     if v) and len(self._hue_samples) >= 2
        do_cluster = (
            enough and
            (self._frame_count == self.SAMPLE_FRAMES or
             self._frame_count % self.RECLUSTER_FRAMES == 0)
        )

        if do_cluster:
            self._last_cluster = self._cluster()

        return dict(self._last_cluster)

    def _sample_torso(self, frame: np.ndarray, p: PlayerDetection) -> Optional[float]:
        """Sample the torso region and return median hue."""
        x1, y1, x2, y2 = p.bbox
        h_frame, w_frame = frame.shape[:2]

        # Try keypoint-guided torso first
        if p.keypoints is not None:
            kp = p.keypoints
            # Shoulders: 5, 6 — Hips: 11, 12
            shoulder_pts = [(kp[i][0], kp[i][1]) for i in (5, 6) if kp[i][2] > 0.35]
            hip_pts      = [(kp[i][0], kp[i][1]) for i in (11,12) if kp[i][2] > 0.35]

            if shoulder_pts and hip_pts:
                sx = int(np.mean([p[0] for p in shoulder_pts]))
                sy = int(np.mean([p[1] for p in shoulder_pts]))
                hx = int(np.mean([p[0] for p in hip_pts]))
                hy = int(np.mean([p[1] for p in hip_pts]))
                # Crop a narrow strip between shoulders and hips
                cx1 = max(0, min(sx, hx) - 15)
                cy1 = max(0, min(sy, hy))
                cx2 = min(w_frame, max(sx, hx) + 15)
                cy2 = min(h_frame, max(sy, hy))
                if cx2 > cx1 + 10 and cy2 > cy1 + 10:
                    roi = frame[cy1:cy2, cx1:cx2]
                    return self._dominant_hue(roi)

        # Fallback: middle vertical third of bounding box
        mh = (y2 - y1) // 3
        crop = frame[
            max(0, y1 + mh): min(h_frame, y2 - mh),
            max(0, x1 + 5):  min(w_frame, x2 - 5),
        ]
        if crop.size == 0:
            return None
        return self._dominant_hue(crop)

    @staticmethod
    def _dominant_hue(roi: np.ndarray) -> Optional[float]:
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Ignore very dark or very desaturated pixels (shadows, black kit)
        mask = cv2.inRange(hsv,
                           np.array([0, 30,  50]),
                           np.array([179, 255, 255]))
        hues = hsv[:, :, 0][mask > 0]
        if len(hues) < 20:
            return None
        return float(np.median(hues))

    def _cluster(self) -> dict[int, str]:
        """
        Assign each player to home or away by finding the largest hue gap.
        Returns {player_id: "home"|"away"}.
        """
        ids   = [pid for pid, samples in self._hue_samples.items() if len(samples) >= 5]
        means = {pid: float(np.mean(list(self._hue_samples[pid]))) for pid in ids}

        if len(means) < 2:
            return {pid: "home" for pid in ids}

        sorted_ids = sorted(means, key=lambda p: means[p])
        hues = [means[p] for p in sorted_ids]

        # Find biggest gap in circular hue space (0-179)
        best_split = 0
        best_gap   = -1
        for i in range(len(hues) - 1):
            gap = hues[i+1] - hues[i]
            if gap > best_gap:
                best_gap   = gap
                best_split = i

        result = {}
        for i, pid in enumerate(sorted_ids):
            result[pid] = "home" if i <= best_split else "away"

        # Keep previously assigned players' teams stable: if both clusters
        # flip compared to last assignment, swap labels
        if self._last_cluster:
            old_home = {p for p, t in self._last_cluster.items() if t == "home"}
            new_home = {p for p, t in result.items()      if t == "home"}
            overlap  = old_home & new_home
            if len(overlap) < len(old_home & set(result)) / 2:
                result = {p: ("away" if t == "home" else "home") for p, t in result.items()}

        return result


# ---------------------------------------------------------------------------
# Vision Pipeline
# ---------------------------------------------------------------------------

class VisionPipeline:
    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.4):
        from ultralytics import YOLO
        print("[Vision] Loading YOLOv8n-pose …")
        self.model = YOLO(model_path)
        self.conf  = conf
        self.shirt_detector = ShirtColourDetector()
        self._ball_history: deque = deque(maxlen=8)
        print("[Vision] Model ready.")

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        team_of: dict,
        draw_poses: bool = True,
        draw_ball:  bool = True,
        pitch_polygon: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[PlayerDetection], Optional[BallDetection], dict[int, str]]:
        """
        Returns:
            annotated       – BGR frame with overlays
            players         – list[PlayerDetection]
            ball            – BallDetection | None
            new_assignments – dict[int, str] shirt-colour team assignments
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # --- Player + pose ---
        results = self.model.track(
            frame, persist=True, conf=self.conf,
            classes=[0], verbose=False, imgsz=640,
        )

        players: list[PlayerDetection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            kps   = results[0].keypoints

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tid = int(box.id[0]) if box.id is not None else i
                kp_data = None
                if kps is not None and i < len(kps.data):
                    kp_data = kps.data[i].cpu().numpy()

                players.append(PlayerDetection(
                    track_id=tid, bbox=(x1, y1, x2, y2),
                    center=(cx, cy), keypoints=kp_data,
                ))

        # --- Shirt colour team detection ---
        new_assignments = self.shirt_detector.update(frame, players)

        # Merge with game's existing assignments (game state takes precedence if set)
        merged = {**new_assignments}
        for p in players:
            if p.track_id in team_of and team_of[p.track_id] is not None:
                merged[p.track_id] = team_of[p.track_id]

        # --- Draw ---
        if draw_poses:
            for p in players:
                team = merged.get(p.track_id)
                self._draw_player(annotated, p, team)

        # --- Ball ---
        ball = self._detect_ball(frame)
        if ball and draw_ball:
            self._draw_ball(annotated, ball)

        # --- Pitch outline ---
        if pitch_polygon is not None:
            cv2.polylines(
                annotated, [pitch_polygon.reshape((-1,1,2))],
                True, (0,255,80), 2, cv2.LINE_AA,
            )

        return annotated, players, ball, new_assignments

    # ------------------------------------------------------------------
    # Ball detection
    # ------------------------------------------------------------------
    def _detect_ball(self, frame: np.ndarray) -> Optional[BallDetection]:
        # YOLO sports-ball
        res = self.model(frame, classes=[SPORTS_BALL_CLASS], conf=0.3,
                         verbose=False, imgsz=640)
        if res and res[0].boxes is not None and len(res[0].boxes):
            box = res[0].boxes[0]
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            r = ((x2-x1)+(y2-y1))/4
            self._ball_history.append((cx, cy))
            return BallDetection((cx,cy), r, float(box.conf[0]), "yolo")

        # HSV fallback
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for rng in BALL_HSV_RANGES:
            mask |= cv2.inRange(hsv, rng["lower"], rng["upper"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (200 < area < 8000): continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            if 4*np.pi*area/(peri**2) < 0.55: continue
            if best is None or area > cv2.contourArea(best):
                best = cnt

        if best is not None:
            (cx,cy), r = cv2.minEnclosingCircle(best)
            self._ball_history.append((int(cx), int(cy)))
            return BallDetection((int(cx),int(cy)), float(r), 0.6, "hsv")

        return None

    def ball_velocity(self, fps: float = 15.0) -> float:
        if len(self._ball_history) < 2:
            return 0.0
        pts = list(self._ball_history)
        n = min(4, len(pts))
        dx = pts[-1][0] - pts[-n][0]
        dy = pts[-1][1] - pts[-n][1]
        return float(np.hypot(dx, dy))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_player(self, frame, p: PlayerDetection, team: Optional[str]):
        x1, y1, x2, y2 = p.bbox
        tid = p.track_id
        colour = TEAM_COLOURS.get(team, TEAM_COLOURS[None])

        # Bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), colour, 2, cv2.LINE_AA)

        # Label with team badge
        team_badge = {"home": "🏠", "away": "✈️", None: "?"}.get(team, "?")
        label = f"P{tid} {team_badge if team else ''}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+6, y1), colour, -1)
        cv2.putText(frame, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2, cv2.LINE_AA)

        if p.keypoints is None:
            return

        # Skeleton
        SKEL_COLS = [
            (255,200,0),(255,200,0),(255,200,0),(255,200,0),
            (0,200,255),(0,200,255),(0,200,255),(0,200,255),(0,200,255),
            (0,255,128),(0,255,128),(0,255,128),
            (255,80,80),(255,80,80),(255,80,80),(255,80,80),
        ]
        for idx, (a,b) in enumerate(SKELETON):
            if a >= len(p.keypoints) or b >= len(p.keypoints): continue
            xa,ya,ca = p.keypoints[a]
            xb,yb,cb = p.keypoints[b]
            if ca < 0.4 or cb < 0.4: continue
            col = SKEL_COLS[min(idx, len(SKEL_COLS)-1)]
            cv2.line(frame, (int(xa),int(ya)), (int(xb),int(yb)), col, 2, cv2.LINE_AA)

        for kp in p.keypoints:
            x,y,c = kp
            if c < 0.4: continue
            cv2.circle(frame, (int(x),int(y)), 4, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x),int(y)), 4, colour,        1,  cv2.LINE_AA)

    def _draw_ball(self, frame, ball: BallDetection):
        cx, cy = ball.center
        r = max(int(ball.radius), 8)
        cv2.circle(frame, (cx,cy), r,   (0,255,255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), r-2, (0,180,180), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 3,   (0,255,255),-1, cv2.LINE_AA)
        cv2.putText(frame, "⚽" if ball.source=="yolo" else "HSV",
                    (cx+r+4, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0,255,255), 1, cv2.LINE_AA)

"""
server.py – Garden Referee main server

Fully autonomous refereeing:
  - First-time finishing detected via ball-speed profiling
  - 3-pass rule triggered by auto dead-ball and auto keeper-pickup
  - Goals confirmed by goal-line technology (trajectory crossing detection)
  - Teams assigned automatically by shirt colour (HSV clustering)

Run:
    python server.py [--camera 0] [--port 5000] [--fps 15] [--width 1280]
"""

import argparse
import json
import threading
import time
import cv2
import numpy as np
from collections import deque
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from vision import VisionPipeline
from game  import GameState, BALL_SPEED_SHOT


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "garden-referee-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
latest_frame: np.ndarray | None = None
frame_lock   = threading.Lock()
var_buffer:   deque = deque(maxlen=450)   # ~15s @ 30fps
var_lock      = threading.Lock()

game:   GameState       = None
vision: VisionPipeline  = None
pitch_polygon: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_pitch_config(path: str = "pitch_config.json"):
    global pitch_polygon
    try:
        with open(path) as f:
            cfg = json.load(f)
        pts = cfg.get("pitch_polygon", [])
        if pts:
            pitch_polygon = np.array(pts, dtype=np.int32)
        print(f"[Server] Pitch polygon: {len(pts)} points")
    except FileNotFoundError:
        print("[Server] No pitch_config.json — run setup_pitch.py first")


# ---------------------------------------------------------------------------
# CV capture + game loop (background thread)
# ---------------------------------------------------------------------------

# Shot debouncing — avoid calling validate_shot 30× per shot
_last_shot_time = 0.0
_SHOT_DEBOUNCE  = 1.5   # seconds

def capture_thread(camera_idx: int, fps: int, width: int):
    global latest_frame, _last_shot_time

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 9 / 16))
    cap.set(cv2.CAP_PROP_FPS, fps)

    interval    = 1.0 / fps
    last_tick   = 0.0
    frame_count = 0

    print(f"[Capture] Camera {camera_idx}, {fps}fps, width={width}")

    while True:
        now = time.time()
        if (now - last_tick) < interval:
            time.sleep(0.005)
            continue
        last_tick = now

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        # ---- Vision pipeline ------------------------------------------
        annotated, players, ball, new_teams = vision.process_frame(
            frame,
            team_of=game.team_of,
            draw_poses=True,
            draw_ball=True,
            pitch_polygon=pitch_polygon,
        )

        # ---- Update game's team assignments ---------------------------
        for pid, team in new_teams.items():
            if game.team_of.get(pid) is None:
                game.assign_team(pid, team)

        ball_speed = vision.ball_velocity(fps)

        # ---- Auto shot detection (first-time rule) --------------------
        now_t = time.time()
        if (ball_speed > BALL_SPEED_SHOT
                and ball is not None
                and game.current_possessor is not None
                and (now_t - _last_shot_time) > _SHOT_DEBOUNCE):
            _last_shot_time = now_t
            toward = _nearest_goal_id(ball, game)
            result = game.validate_shot(game.current_possessor, toward)
            if not result["valid"]:
                socketio.emit("foul", result)
                socketio.emit("game_state", game.get_state())

        # ---- Game tick -----------------------------------------------
        event = game.update(players, ball, ball_speed)

        # ---- HUD overlay ---------------------------------------------
        _draw_hud(annotated, game, ball_speed, fps)

        # ---- Broadcast events ----------------------------------------
        if event:
            state = game.get_state()
            socketio.emit("game_state", state)

            if event.startswith("goal_"):
                gid = int(event[-1])
                socketio.emit("goal", {
                    "goal_id": gid,
                    "score":   game.score,
                    "message": game.commentary[-1]["text"] if game.commentary else "",
                })
            elif event in ("dead_ball", "keeper_pickup"):
                socketio.emit("dead_ball_event", {
                    "trigger":  event,
                    "message":  game.commentary[-1]["text"] if game.commentary else "",
                })
            elif event == "pass":
                socketio.emit("pass_event", {
                    "pass_count": game.pass_count,
                    "required":   game.required_passes,
                    "phase":      game.phase,
                    "message":    game.commentary[-1]["text"] if game.commentary else "",
                })

        elif frame_count % 12 == 0:
            socketio.emit("game_state", game.get_state())

        # ---- Store frame ---------------------------------------------
        with frame_lock:
            latest_frame = annotated

        ret2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret2:
            with var_lock:
                var_buffer.append(buf.tobytes())

    cap.release()


def _nearest_goal_id(ball, gs: GameState) -> int:
    if not gs.goals:
        return 0
    bx, by = ball.center
    return min(gs.goals, key=lambda g: (
        (g.center()[0]-bx)**2 + (g.center()[1]-by)**2
    )).id


# ---------------------------------------------------------------------------
# HUD — drawn on frame in capture thread
# ---------------------------------------------------------------------------

def _draw_hud(frame: np.ndarray, gs: GameState, ball_speed: float, fps: int):
    h, w = frame.shape[:2]

    # ---- Score (centre-top) ------------------------------------------
    score_str = f"{gs.score['home']}  -  {gs.score['away']}"
    (sw, sh), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    sx = (w - sw) // 2
    cv2.rectangle(frame, (sx-12, 6), (sx+sw+12, 6+sh+14), (0,0,0), -1)
    cv2.putText(frame, score_str, (sx, 6+sh+4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # ---- Pass counter bar (top-left) ---------------------------------
    bar_y, bar_x = 10, 12
    bar_w, bar_h = 180, 22
    fill = int(bar_w * min(gs.pass_count, gs.required_passes) / gs.required_passes)
    col  = (0,200,80) if gs.pass_count >= gs.required_passes else (0,120,220)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (20,20,20), -1)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), 1)
    label = f"PASSES  {gs.pass_count}/{gs.required_passes}"
    cv2.putText(frame, label, (bar_x+6, bar_y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    # ---- Phase badge (top-right) ------------------------------------
    phase_map = {"live": ("LIVE", (0,180,60)), "dead_ball": ("DEAD BALL", (0,120,230)),
                 "kickoff": ("KICKOFF", (180,160,0))}
    phase_txt, phase_col = phase_map.get(gs.phase, ("UNKNOWN", (120,120,120)))
    (pw,_),_ = cv2.getTextSize(phase_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    px = w - pw - 24
    cv2.rectangle(frame, (px-8, 8), (px+pw+8, 8+22), (20,20,20), -1)
    cv2.putText(frame, phase_txt, (px, 8+17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_col, 2, cv2.LINE_AA)

    # ---- VAR border --------------------------------------------------
    if gs.var_active:
        cv2.rectangle(frame, (0,0), (w-1, h-1), (0,200,255), 5)
        cv2.putText(frame, "VAR", (w//2-35, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,220,255), 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/var_feed")
def var_feed():
    return Response(_var_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(game.get_state())


# ----- Manual overrides (referee can disagree with AI) --------------------

@app.route("/api/override/keeper_pickup", methods=["POST"])
def api_keeper_pickup():
    result = game.manual_keeper_pickup()
    socketio.emit("game_state", game.get_state())
    socketio.emit("dead_ball_event", {"trigger": "keeper_pickup", "message": result["message"]})
    return jsonify(result)


@app.route("/api/override/dead_ball", methods=["POST"])
def api_dead_ball():
    result = game.manual_dead_ball()
    socketio.emit("game_state", game.get_state())
    socketio.emit("dead_ball_event", {"trigger": "dead_ball", "message": result["message"]})
    return jsonify(result)


@app.route("/api/override/goal/<int:goal_id>", methods=["POST"])
def api_goal(goal_id):
    result = game.manual_goal(goal_id)
    socketio.emit("game_state", game.get_state())
    socketio.emit("goal", {"goal_id": goal_id, "score": result["score"],
                            "message": result["message"]})
    return jsonify(result)


@app.route("/api/override/disallow_goal", methods=["POST"])
def api_disallow():
    # Remove the last goal
    if game.last_goal is not None:
        scorer = "away" if game.last_goal == 0 else "home"
        if game.score[scorer] > 0:
            game.score[scorer] -= 1
        msg = "Goal disallowed by referee override."
        game._add_raw(msg)
        game.phase = "dead_ball"
        socketio.emit("game_state", game.get_state())
        socketio.emit("goal_disallowed", {"message": msg})
        return jsonify({"message": msg})
    return jsonify({"message": "No recent goal to disallow."})


@app.route("/api/var/start", methods=["POST"])
def api_var_start():
    result = game.trigger_var()
    socketio.emit("game_state", game.get_state())
    socketio.emit("var_start", result)
    return jsonify(result)


@app.route("/api/var/cancel", methods=["POST"])
def api_var_cancel():
    result = game.cancel_var()
    socketio.emit("game_state", game.get_state())
    socketio.emit("var_end", result)
    return jsonify(result)


# ---------------------------------------------------------------------------
# MJPEG generators
# ---------------------------------------------------------------------------

def _mjpeg_stream():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


def _var_stream():
    with var_lock:
        frames = list(var_buffer)
    if not frames:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
        return
    for fb in frames:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n")
        time.sleep(0.05)
    while game and game.var_active:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frames[-1] + b"\r\n")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global game, vision

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int,   default=0)
    parser.add_argument("--port",   type=int,   default=5000)
    parser.add_argument("--fps",    type=int,   default=15)
    parser.add_argument("--width",  type=int,   default=1280)
    args = parser.parse_args()

    load_pitch_config()
    vision = VisionPipeline()
    game   = GameState()

    t = threading.Thread(
        target=capture_thread,
        args=(args.camera, args.fps, args.width),
        daemon=True,
    )
    t.start()

    print(f"\n  Garden Referee running at http://localhost:{args.port}\n")
    socketio.run(app, host="0.0.0.0", port=args.port,
                 debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

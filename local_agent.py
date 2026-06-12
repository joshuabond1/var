"""
local_agent.py – Run this on the laptop at the pitch
------------------------------------------------------
Captures webcam, runs the full CV + game engine, then pushes processed
frames and game events to the Railway relay server so remote viewers
can watch and see live referee decisions.

Usage:
    python local_agent.py [--camera 0] [--fps 15] [--width 1280]

Environment variables (put in .env or export before running):
    RELAY_URL      https://your-app.railway.app   (required)
    RELAY_SECRET   your-shared-secret             (must match Railway)

The script also serves localhost:5001 for your own live view on the
pitch laptop — you don't need to watch through Railway yourself.
"""

import argparse
import json
import os
import threading
import time
import cv2
import numpy as np
from collections import deque
from dotenv import load_dotenv
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO
import socketio as sio_client   # python-socketio client

from vision import VisionPipeline
from game   import GameState, BALL_SPEED_SHOT

load_dotenv()

RELAY_URL    = os.environ.get("RELAY_URL",    "")
RELAY_SECRET = os.environ.get("RELAY_SECRET", "")

# ---------------------------------------------------------------------------
# Local Flask app (for the pitch-side laptop view)
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "garden-referee-local"
local_socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
latest_frame: np.ndarray | None = None
frame_lock   = threading.Lock()
var_buffer: deque = deque(maxlen=450)

game:   GameState       = None
vision: VisionPipeline  = None

# Relay SocketIO client
relay_client: sio_client.Client | None = None
relay_connected = False

_last_shot_time = 0.0
_SHOT_DEBOUNCE  = 1.5


# ---------------------------------------------------------------------------
# Relay connection (background thread)
# ---------------------------------------------------------------------------

def connect_to_relay():
    global relay_client, relay_connected

    if not RELAY_URL:
        print("[Agent] No RELAY_URL set — running in local-only mode (localhost:5001)")
        return

    client = sio_client.Client(reconnection=True, reconnection_attempts=0,
                                reconnection_delay=3)

    @client.event(namespace="/agent")
    def connect():
        global relay_connected
        relay_connected = True
        print(f"[Agent] Connected to relay at {RELAY_URL}")
        client.emit("auth", {"secret": RELAY_SECRET}, namespace="/agent")

    @client.on("auth_result", namespace="/agent")
    def on_auth(data):
        if data.get("ok"):
            print("[Agent] Relay auth OK — streaming frames")
        else:
            print(f"[Agent] Relay auth FAILED: {data.get('reason')} — check RELAY_SECRET")

    @client.event(namespace="/agent")
    def disconnect():
        global relay_connected
        relay_connected = False
        print("[Agent] Disconnected from relay — will reconnect automatically")

    try:
        client.connect(RELAY_URL, namespaces=["/agent"])
        relay_client = client
    except Exception as e:
        print(f"[Agent] Could not connect to relay: {e}")


def push_to_relay(event: str, data):
    """Emit an event to the relay. Fire-and-forget."""
    if relay_client and relay_connected:
        try:
            relay_client.emit(event, data, namespace="/agent")
        except Exception:
            pass


def push_frame_to_relay(jpeg_bytes: bytes):
    if relay_client and relay_connected:
        try:
            relay_client.emit("frame", jpeg_bytes, namespace="/agent")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pitch config
# ---------------------------------------------------------------------------
def load_pitch_config(path: str = "pitch_config.json"):
    try:
        with open(path) as f:
            cfg = json.load(f)
        pts = cfg.get("pitch_polygon", [])
        return np.array(pts, dtype=np.int32) if pts else None
    except FileNotFoundError:
        print("[Agent] No pitch_config.json — run setup_pitch.py first")
        return None


# ---------------------------------------------------------------------------
# CV + game loop (background thread)
# ---------------------------------------------------------------------------

def capture_thread(camera_idx: int, fps: int, width: int, pitch_polygon):
    global latest_frame, _last_shot_time

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 9 / 16))
    cap.set(cv2.CAP_PROP_FPS, fps)

    interval    = 1.0 / fps
    last_tick   = 0.0
    frame_count = 0

    # Push frames to relay at a lower rate to save upload bandwidth
    relay_fps       = 12
    relay_interval  = 1.0 / relay_fps
    last_relay_push = 0.0

    print(f"[Agent] Capture: camera={camera_idx}, {fps}fps, width={width}")

    while True:
        now = time.time()
        if (now - last_tick) < interval:
            time.sleep(0.005)
            continue
        last_tick = now
        frame_count += 1

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # --- CV pipeline ---
        annotated, players, ball, new_teams = vision.process_frame(
            frame,
            team_of=game.team_of,
            draw_poses=True,
            draw_ball=True,
            pitch_polygon=pitch_polygon,
        )

        for pid, team in new_teams.items():
            if game.team_of.get(pid) is None:
                game.assign_team(pid, team)

        ball_speed = vision.ball_velocity(fps)

        # --- Auto shot / first-time rule ---
        if (ball_speed > BALL_SPEED_SHOT
                and ball is not None
                and game.current_possessor is not None
                and (now - _last_shot_time) > _SHOT_DEBOUNCE):
            _last_shot_time = now
            toward = _nearest_goal(ball, game)
            result = game.validate_shot(game.current_possessor, toward)
            if not result["valid"]:
                _broadcast("foul", result)

        # --- Game tick ---
        event = game.update(players, ball, ball_speed)
        _draw_hud(annotated, game, ball_speed)

        if event:
            state = game.get_state()
            _broadcast("game_state", state)
            if event.startswith("goal_"):
                _broadcast("goal", {
                    "goal_id": int(event[-1]),
                    "score":   game.score,
                    "message": game.commentary[-1]["text"] if game.commentary else "",
                })
            elif event in ("dead_ball", "keeper_pickup"):
                _broadcast("dead_ball_event", {
                    "trigger": event,
                    "message": game.commentary[-1]["text"] if game.commentary else "",
                })
            elif event == "pass":
                _broadcast("pass_event", {
                    "pass_count": game.pass_count,
                    "required":   game.required_passes,
                    "phase":      game.phase,
                    "message":    game.commentary[-1]["text"] if game.commentary else "",
                })
        elif frame_count % 12 == 0:
            _broadcast("game_state", game.get_state())

        # --- Store frame locally ---
        with frame_lock:
            latest_frame = annotated

        # --- Push to relay ---
        if (now - last_relay_push) >= relay_interval:
            last_relay_push = now
            # Slightly lower quality for network efficiency
            ret2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ret2:
                fb = buf.tobytes()
                push_frame_to_relay(fb)
                with var_buffer.mutex if hasattr(var_buffer, 'mutex') else open(os.devnull):
                    pass
                var_buffer.append(fb)

    cap.release()


def _broadcast(event: str, data):
    """Emit to local viewers AND push to relay."""
    local_socketio.emit(event, data)
    push_to_relay(event, data)


def _nearest_goal(ball, gs) -> int:
    if not gs.goals:
        return 0
    bx, by = ball.center
    return min(gs.goals, key=lambda g: (
        (g.center()[0]-bx)**2 + (g.center()[1]-by)**2
    )).id


# ---------------------------------------------------------------------------
# HUD overlay (same as server.py)
# ---------------------------------------------------------------------------

def _draw_hud(frame, gs, ball_speed):
    h, w = frame.shape[:2]
    score_str = f"{gs.score['home']}  -  {gs.score['away']}"
    (sw, sh), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    sx = (w - sw) // 2
    cv2.rectangle(frame, (sx-12, 6), (sx+sw+12, 6+sh+14), (0,0,0), -1)
    cv2.putText(frame, score_str, (sx, 6+sh+4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    bar_y, bar_x = 10, 12
    bar_w, bar_h = 180, 22
    fill = int(bar_w * min(gs.pass_count, gs.required_passes) / gs.required_passes)
    col  = (0,200,80) if gs.pass_count >= gs.required_passes else (0,120,220)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (20,20,20), -1)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), 1)
    cv2.putText(frame, f"PASSES  {gs.pass_count}/{gs.required_passes}",
                (bar_x+6, bar_y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    phase_map = {"live": ("LIVE",(0,180,60)), "dead_ball": ("DEAD BALL",(0,120,230)),
                 "kickoff": ("KICKOFF",(180,160,0))}
    phase_txt, phase_col = phase_map.get(gs.phase, ("UNKNOWN",(120,120,120)))
    (pw,_),_ = cv2.getTextSize(phase_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    px = w - pw - 24
    cv2.rectangle(frame, (px-8, 8), (px+pw+8, 8+22), (20,20,20), -1)
    cv2.putText(frame, phase_txt, (px, 8+17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_col, 2, cv2.LINE_AA)

    if gs.var_active:
        cv2.rectangle(frame, (0,0), (w-1, h-1), (0,200,255), 5)
        cv2.putText(frame, "VAR", (w//2-35, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,220,255), 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Local Flask routes (pitch-side laptop view on :5001)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_local(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/var_feed")
def var_feed():
    return Response(_var_local(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    return jsonify(game.get_state())

@app.route("/api/override/keeper_pickup", methods=["POST"])
def api_keeper():
    result = game.manual_keeper_pickup()
    _broadcast("game_state", game.get_state())
    _broadcast("dead_ball_event", {"trigger": "keeper_pickup", "message": result["message"]})
    return jsonify(result)

@app.route("/api/override/dead_ball", methods=["POST"])
def api_dead():
    result = game.manual_dead_ball()
    _broadcast("game_state", game.get_state())
    _broadcast("dead_ball_event", {"trigger": "dead_ball", "message": result["message"]})
    return jsonify(result)

@app.route("/api/override/goal/<int:goal_id>", methods=["POST"])
def api_goal(goal_id):
    result = game.manual_goal(goal_id)
    _broadcast("game_state", game.get_state())
    _broadcast("goal", {"goal_id": goal_id, "score": result["score"], "message": result["message"]})
    return jsonify(result)

@app.route("/api/override/disallow_goal", methods=["POST"])
def api_disallow():
    scorer = "away" if game.last_goal == 0 else "home"
    if game.last_goal is not None and game.score[scorer] > 0:
        game.score[scorer] -= 1
        msg = "Goal disallowed by referee override."
        game._add_raw(msg)
        game.phase = "dead_ball"
        _broadcast("game_state", game.get_state())
        _broadcast("goal_disallowed", {"message": msg})
        return jsonify({"message": msg})
    return jsonify({"message": "No recent goal."})

@app.route("/api/var/start", methods=["POST"])
def api_var_start():
    result = game.trigger_var()
    _broadcast("game_state", game.get_state())
    _broadcast("var_start", result)
    return jsonify(result)

@app.route("/api/var/cancel", methods=["POST"])
def api_var_cancel():
    result = game.cancel_var()
    _broadcast("game_state", game.get_state())
    _broadcast("var_end", result)
    return jsonify(result)


def _mjpeg_local():
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


def _var_local():
    frames = list(var_buffer)
    if not frames:
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
    parser.add_argument("--fps",    type=int,   default=15)
    parser.add_argument("--width",  type=int,   default=1280)
    parser.add_argument("--port",   type=int,   default=5001)
    args = parser.parse_args()

    pitch_polygon = load_pitch_config()
    vision = VisionPipeline()
    game   = GameState()

    # Connect to Railway relay in background
    threading.Thread(target=connect_to_relay, daemon=True).start()

    # Start capture + CV loop
    threading.Thread(
        target=capture_thread,
        args=(args.camera, args.fps, args.width, pitch_polygon),
        daemon=True,
    ).start()

    relay_status = f" -> pushing to {RELAY_URL}" if RELAY_URL else " (local only — set RELAY_URL to stream remotely)"
    print(f"\n  Pitch-side view: http://localhost:{args.port}")
    print(f"  Relay status:   {relay_status}\n")

    local_socketio.run(app, host="0.0.0.0", port=args.port,
                       debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

"""
relay_server.py – Deploy this to Railway
-----------------------------------------
Receives processed video frames and game events from the local laptop agent,
then serves them to any number of remote viewers.

No webcam, no CV, no heavy models — just a lightweight relay.

Architecture:
    Laptop (local_agent.py)  ──SocketIO──>  Railway (relay_server.py)  ──browser
                                /agent namespace                         / namespace

Environment variables (set in Railway dashboard):
    RELAY_SECRET   A shared secret string — must match the laptop's .env
    PORT           Set automatically by Railway — do not set manually
"""

from gevent import monkey; monkey.patch_all()

import os
import time
import threading
from collections import deque
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit, join_room

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("RELAY_SECRET", "garden-referee-relay")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

RELAY_SECRET = os.environ.get("RELAY_SECRET", "")

# ---------------------------------------------------------------------------
# Shared state (in-memory — fine for a single Railway instance)
# ---------------------------------------------------------------------------
latest_frame_bytes: bytes | None = None
frame_lock = threading.Lock()

var_buffer: deque = deque(maxlen=450)
var_lock    = threading.Lock()

_game_state: dict = {
    "score":             {"home": 0, "away": 0},
    "phase":             "dead_ball",
    "pass_count":        0,
    "required_passes":   3,
    "current_possessor": None,
    "possessor_team":    None,
    "team_of":           {},
    "last_foul":         None,
    "last_goal":         None,
    "var_active":        False,
    "commentary":        [],
}
_state_lock = threading.Lock()

_last_frame_ts = 0.0   # to detect agent disconnect


# ---------------------------------------------------------------------------
# Agent namespace — local laptop connects here to push data
# ---------------------------------------------------------------------------

@socketio.on("connect", namespace="/agent")
def agent_connect():
    token = os.environ.get("RELAY_SECRET", "")
    # Token is passed as the first message after connect; we do a simple
    # handshake in on_auth below rather than blocking the connect event.
    print("[Relay] Agent connected")


@socketio.on("auth", namespace="/agent")
def on_auth(data):
    secret = RELAY_SECRET
    if secret and data.get("secret") != secret:
        emit("auth_result", {"ok": False, "reason": "bad secret"})
        return
    emit("auth_result", {"ok": True})
    print("[Relay] Agent authenticated")


@socketio.on("frame", namespace="/agent")
def on_frame(data):
    """Receive a JPEG frame (bytes) from the local agent."""
    global latest_frame_bytes, _last_frame_ts
    fb = data if isinstance(data, bytes) else data.get("frame", b"")
    with frame_lock:
        latest_frame_bytes = fb
        _last_frame_ts = time.time()
    with var_lock:
        var_buffer.append(fb)


@socketio.on("game_state", namespace="/agent")
def on_game_state(state: dict):
    with _state_lock:
        _game_state.update(state)
    # Forward to all viewers
    socketio.emit("game_state", state, namespace="/")


@socketio.on("goal", namespace="/agent")
def on_goal(data):
    socketio.emit("goal", data, namespace="/")


@socketio.on("foul", namespace="/agent")
def on_foul(data):
    socketio.emit("foul", data, namespace="/")


@socketio.on("dead_ball_event", namespace="/agent")
def on_dead_ball(data):
    socketio.emit("dead_ball_event", data, namespace="/")


@socketio.on("pass_event", namespace="/agent")
def on_pass(data):
    socketio.emit("pass_event", data, namespace="/")


@socketio.on("var_start", namespace="/agent")
def on_var_start(data):
    socketio.emit("var_start", data, namespace="/")


@socketio.on("var_end", namespace="/agent")
def on_var_end(data):
    socketio.emit("var_end", data, namespace="/")


@socketio.on("goal_disallowed", namespace="/agent")
def on_goal_disallowed(data):
    socketio.emit("goal_disallowed", data, namespace="/")


# ---------------------------------------------------------------------------
# Viewer namespace
# ---------------------------------------------------------------------------

@socketio.on("connect", namespace="/")
def viewer_connect():
    with _state_lock:
        state = dict(_game_state)
    emit("game_state", state)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/var_feed")
def var_feed():
    return Response(
        _var_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/state")
def api_state():
    with _state_lock:
        return jsonify(_game_state)


@app.route("/api/status")
def api_status():
    """Health check — also reports whether the agent is connected."""
    age = time.time() - _last_frame_ts
    return jsonify({
        "relay": "ok",
        "agent_connected": age < 5.0,
        "last_frame_age_s": round(age, 1),
    })


# ---------------------------------------------------------------------------
# MJPEG generators
# ---------------------------------------------------------------------------

_WAITING_JPEG: bytes | None = None  # generated once as a placeholder


def _get_waiting_frame() -> bytes:
    """Return a tiny placeholder JPEG when no agent is connected."""
    global _WAITING_JPEG
    if _WAITING_JPEG is None:
        import numpy as np
        import cv2
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (14, 21, 32)
        cv2.putText(img, "WAITING FOR MATCH FEED",
                    (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (50, 80, 120), 2, cv2.LINE_AA)
        cv2.putText(img, "Start local_agent.py on the pitch laptop",
                    (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (40, 60, 90), 1, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", img)
        _WAITING_JPEG = buf.tobytes()
    return _WAITING_JPEG


def _mjpeg_stream():
    while True:
        with frame_lock:
            fb = latest_frame_bytes
        if fb is None:
            fb = _get_waiting_frame()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n")
        time.sleep(0.04)  # ~25fps cap for remote viewers


def _var_stream():
    with var_lock:
        frames = list(var_buffer)
    if not frames:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + _get_waiting_frame() + b"\r\n")
        return
    for fb in frames:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n")
        time.sleep(0.06)
    # Freeze on last frame
    while _game_state.get("var_active"):
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frames[-1] + b"\r\n")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[Relay] Running on port {port}")
    socketio.run(app, host="0.0.0.0", port=port,
                 debug=False, use_reloader=False,
                 allow_unsafe_werkzeug=True)

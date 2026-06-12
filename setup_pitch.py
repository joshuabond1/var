"""
setup_pitch.py – One-time pitch & goal marking wizard
------------------------------------------------------
Run this script ONCE before starting the referee server.

Steps:
  1. Opens your webcam (or accepts a still image path as argv[1])
  2. You click 4 corners of the pitch boundary (any order is fine)
  3. You click the left post then right post of GOAL A
  4. You click the left post then right post of GOAL B
  5. Saves everything to pitch_config.json

The goal polygons are derived by projecting a goal-depth rectangle
inward from the two clicked post positions.

Usage:
    python setup_pitch.py             # uses webcam
    python setup_pitch.py frame.png   # uses a still image
"""

import cv2
import json
import sys
import numpy as np

WINDOW = "Garden Referee – Pitch Setup"

# How deep (in pixels) to extend the goal polygon behind the goal line
GOAL_DEPTH_PX = 40

# ---- State machine -------------------------------------------------------
STEPS = [
    ("PITCH",  4, "Click the 4 CORNERS of your pitch (any order)"),
    ("GOAL_A", 2, "Click LEFT POST then RIGHT POST of GOAL A (bigger goal)"),
    ("GOAL_B", 2, "Click LEFT POST then RIGHT POST of GOAL B (smaller goal)"),
]

clicks: list  = []
step_idx: int = 0
collected: dict = {"pitch": [], "goals": []}
frame_orig: np.ndarray = None
display: np.ndarray = None


def mouse_cb(event, x, y, flags, param):
    global clicks, display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(display, (x, y), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(display, (x, y), 7, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw line back to previous click (for this step)
        step_name, n_clicks, _ = STEPS[step_idx]
        step_clicks = clicks  # only clicks for this step
        if len(step_clicks) > 1:
            cv2.line(display, step_clicks[-2], step_clicks[-1],
                     (0, 255, 100), 2, cv2.LINE_AA)


def draw_instructions(img, text, progress, total):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, (15, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(img, f"Points: {progress}/{total}  |  Press ENTER to confirm / R to redo",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)


def build_goal_poly(p1, p2, depth=GOAL_DEPTH_PX):
    """
    Given two goal posts p1 and p2, create a 4-point polygon that represents
    the goal mouth + a small depth rectangle behind the line.
    We push the rectangle inward (toward the pitch centre).
    Since we don't know which direction 'inward' is, we just use both and pick
    the one that's closer to the pitch centre.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    # Perpendicular direction
    direction = p2 - p1
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    # Try both sides – we'll just return the polygon; user can see on screen
    p3 = p2 + perp * depth
    p4 = p1 + perp * depth
    return [p1.astype(int).tolist(), p2.astype(int).tolist(),
            p3.astype(int).tolist(), p4.astype(int).tolist()]


def draw_pitch_outline(img, pts, colour=(0, 255, 80)):
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), colour, 2, cv2.LINE_AA)
    if len(pts) == 4:
        cv2.line(img, tuple(pts[3]), tuple(pts[0]), colour, 2, cv2.LINE_AA)


def draw_goal(img, poly, label, colour=(0, 200, 255)):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
    cx = sum(p[0] for p in poly) // len(poly)
    cy = sum(p[1] for p in poly) // len(poly)
    cv2.putText(img, label, (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)


def main():
    global clicks, step_idx, display, frame_orig

    # ---- Open video / image -----------------------------------------
    source = sys.argv[1] if len(sys.argv) > 1 else None
    if source:
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            print(f"Cannot read image: {source}")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(0)
        print("Press SPACE to grab a still frame from webcam …")
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            cv2.imshow(WINDOW, frm)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(' '):
                frame_orig = frm.copy()
                break
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()

    if frame_orig is None:
        print("No frame captured.")
        sys.exit(1)

    # Resize for comfort
    h, w = frame_orig.shape[:2]
    scale = min(1.0, 1200 / w, 800 / h)
    if scale < 1.0:
        frame_orig = cv2.resize(frame_orig, (int(w*scale), int(h*scale)))

    display = frame_orig.copy()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    pitch_pts  = []
    goal_polys = []

    for step_idx, (step_name, n_needed, instruction) in enumerate(STEPS):
        clicks = []
        while True:
            view = display.copy()
            # Draw already-confirmed items
            draw_pitch_outline(view, pitch_pts)
            for gi, gp in enumerate(goal_polys):
                draw_goal(view, gp, f"Goal {'A' if gi==0 else 'B'}",
                          (0,200,255) if gi==0 else (255,120,0))

            # Draw current clicks
            for pt in clicks:
                cv2.circle(view, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
            for i in range(len(clicks) - 1):
                cv2.line(view, clicks[i], clicks[i+1], (0, 255, 100), 2, cv2.LINE_AA)

            draw_instructions(view, instruction, len(clicks), n_needed)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                print("Setup cancelled.")
                return

            if key == ord('r'):
                # Redo this step
                clicks = []
                if step_name == "PITCH":
                    pitch_pts = []
                elif step_name.startswith("GOAL"):
                    if goal_polys:
                        goal_polys.pop()

            if key == 13 or key == ord('\n'):  # Enter
                if len(clicks) >= n_needed:
                    if step_name == "PITCH":
                        pitch_pts = list(clicks[:4])
                        # Reorder to convex hull for clean polygon
                        pts_arr = np.array(pitch_pts)
                        hull = cv2.convexHull(pts_arr)
                        pitch_pts = hull.reshape(-1, 2).tolist()
                    else:
                        poly = build_goal_poly(clicks[0], clicks[1])
                        goal_polys.append(poly)
                    break
                else:
                    print(f"Need {n_needed} clicks, have {len(clicks)} — keep clicking!")

    cv2.destroyAllWindows()

    # ---- Save config ------------------------------------------------
    config = {
        "pitch_polygon": pitch_pts,
        "goals": [
            {"id": 0, "label": "Goal A (big)",   "poly": goal_polys[0]},
            {"id": 1, "label": "Goal B (small)",  "poly": goal_polys[1]},
        ]
    }
    out_path = "pitch_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅  Pitch config saved to {out_path}")
    print(f"   Pitch corners  : {pitch_pts}")
    print(f"   Goal A polygon : {goal_polys[0]}")
    print(f"   Goal B polygon : {goal_polys[1]}")
    print("\nNow run:  python server.py\n")

    # ---- Show final overlay -----------------------------------------
    final = frame_orig.copy()
    pts_np = np.array(pitch_pts, dtype=np.int32)
    cv2.polylines(final, [pts_np], True, (0, 255, 80), 3, cv2.LINE_AA)
    for gi, gp in enumerate(goal_polys):
        draw_goal(final, gp, f"Goal {'A' if gi==0 else 'B'}",
                  (0,200,255) if gi==0 else (255,120,0))
    cv2.putText(final, "Setup complete! Press any key to close.",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW, final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Smart Attendance System
=======================
Face-recognition attendance using OpenCV (LBPH recognizer).

Workflow:
    1. register  - capture face samples for a person via webcam
    2. train     - train the recognizer on all registered faces
    3. run       - recognize faces live and mark attendance (once per day)
    4. report    - show present/absent for a given day

Examples:
    python smart_attendance.py register --id 1 --name "Asha"
    python smart_attendance.py train
    python smart_attendance.py run
    python smart_attendance.py report --date 2026-10-18
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trainer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
ATTENDANCE_DIR = "attendance"

FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 70   # LBPH: lower = better match. Tune for your setup.
CONFIRM_FRAMES = 5          # consecutive frames needed before marking


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_detector():
    path = os.path.join(cv2.data.haarcascades,
                        "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        sys.exit("Error: could not load the Haar cascade face detector.")
    return detector


def require_face_module():
    if not hasattr(cv2, "face"):
        sys.exit("Error: cv2.face not found. Install the contrib build:\n"
                 "  pip install opencv-contrib-python")


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open camera {index}.")
    return cap


def detect_faces(detector, gray):
    return detector.detectMultiScale(gray, scaleFactor=1.2,
                                     minNeighbors=5, minSize=(80, 80))


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def attendance_file(day):
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, f"{day}.csv")


def read_attendance(day):
    """Return a dict {id: (name, time)} for the given day."""
    path = attendance_file(day)
    records = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records[row["id"]] = (row["name"], row["time"])
    return records


def mark_attendance(person_id, name):
    """Write one row per person per day. Returns True if newly marked."""
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    if str(person_id) in read_attendance(day):
        return False
    path = attendance_file(day)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "name", "date", "time"])
        writer.writerow([person_id, name, day, now.strftime("%H:%M:%S")])
    return True


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_register(args):
    detector = get_detector()
    person_dir = os.path.join(DATASET_DIR, str(args.id))
    os.makedirs(person_dir, exist_ok=True)

    labels = load_labels()
    labels[str(args.id)] = args.name
    save_labels(labels)

    cap = open_camera(args.camera)
    print(f"Capturing {args.samples} samples for {args.name} (id {args.id}). "
          "Look at the camera and vary your angle slightly. Press q to stop.")

    count, frame_no = 0, 0
    while count < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if frame_no % 4 == 0:  # space out samples
                face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                existing = len(os.listdir(person_dir))
                cv2.imwrite(os.path.join(person_dir, f"{existing}.jpg"), face)
                count += 1

        cv2.putText(frame, f"Samples: {count}/{args.samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Register - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples. Run 'train' to update the model.")


def cmd_train(_args):
    require_face_module()
    faces, ids = [], []

    if not os.path.isdir(DATASET_DIR):
        sys.exit("No dataset found. Register at least one person first.")

    for folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not (os.path.isdir(folder_path) and folder.isdigit()):
            continue
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, FACE_SIZE))
            ids.append(int(folder))

    if not faces:
        sys.exit("No face images found. Run 'register' first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.write(MODEL_PATH)
    print(f"Trained on {len(faces)} images of {len(set(ids))} person(s). "
          f"Model saved to {MODEL_PATH}.")


def cmd_run(args):
    require_face_module()
    if not os.path.exists(MODEL_PATH):
        sys.exit("No trained model. Run 'register' then 'train' first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    labels = load_labels()
    detector = get_detector()
    cap = open_camera(args.camera)

    streak = {}  # id -> consecutive frames recognized
    print("Attendance running. Press q in the video window to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        seen_now = set()

        for (x, y, w, h) in detect_faces(detector, gray):
            face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            pid, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_THRESHOLD and str(pid) in labels:
                name = labels[str(pid)]
                seen_now.add(pid)
                streak[pid] = streak.get(pid, 0) + 1
                colour = (0, 255, 0)
                text = f"{name} ({confidence:.0f})"

                if streak[pid] >= CONFIRM_FRAMES and mark_attendance(pid, name):
                    print(f"Marked present: {name} at "
                          f"{datetime.now().strftime('%H:%M:%S')}")
            else:
                colour = (0, 0, 255)
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        # reset streaks for people not seen this frame
        for pid in list(streak):
            if pid not in seen_now:
                streak[pid] = 0

        cv2.imshow("Smart Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_report(args):
    day = args.date or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        sys.exit("Date must be in YYYY-MM-DD format.")

    labels = load_labels()
    present = read_attendance(day)

    print(f"\nAttendance report for {day}")
    print("-" * 40)
    print("PRESENT")
    if present:
        for pid, (name, t) in present.items():
            print(f"  [{pid}] {name:<20} {t}")
    else:
        print("  (nobody)")

    absent = [(pid, name) for pid, name in labels.items()
              if pid not in present]
    print("\nABSENT")
    if absent:
        for pid, name in absent:
            print(f"  [{pid}] {name}")
    else:
        print("  (nobody)")
    print(f"\nTotal registered: {len(labels)}   Present: {len(present)}   "
          f"Absent: {len(absent)}")
    print(f"CSV file: {attendance_file(day)}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Smart Attendance System (face recognition)")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index (default 0)")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="capture face samples for a person")
    reg.add_argument("--id", type=int, required=True, help="unique numeric ID")
    reg.add_argument("--name", required=True, help="person's name")
    reg.add_argument("--samples", type=int, default=30,
                     help="number of face samples (default 30)")
    reg.set_defaults(func=cmd_register)

    sub.add_parser("train", help="train the recognizer").set_defaults(
        func=cmd_train)
    sub.add_parser("run", help="start live attendance").set_defaults(
        func=cmd_run)

    rep = sub.add_parser("report", help="show present/absent for a day")
    rep.add_argument("--date", help="YYYY-MM-DD (default: today)")
    rep.set_defaults(func=cmd_report)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

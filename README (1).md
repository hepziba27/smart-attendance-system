# Smart Attendance System

Automated, contactless attendance using real-time face recognition. Point a camera at a room, and the system detects and identifies registered people, then logs each person's attendance with a date and timestamp.

## Features

- Register people in seconds with a webcam
- Real-time face detection and recognition (OpenCV, LBPH recognizer)
- Attendance marked once per person per day, with no duplicate entries
- Requires several consecutive matching frames before marking, which reduces false positives
- Daily CSV files that open in Excel or Google Sheets
- Present and absent report for any date
- Single Python file, small dependency list

## Requirements

- Python 3.8+
- A webcam
- `opencv-contrib-python` and `numpy`

## Installation

```bash
git clone https://github.com/<your-username>/smart-attendance.git
cd smart-attendance
pip install opencv-contrib-python numpy
```

Use `opencv-contrib-python`, not plain `opencv-python`. The face recognizer lives in the contrib build.

## Usage

### 1. Register a person

```bash
python smart_attendance.py register --id 1 --name "Asha"
```

Look at the camera and move your head slightly while it captures 30 samples. Use `--samples` to change the number. Repeat for each person, with a unique numeric ID each time.

### 2. Train the model

```bash
python smart_attendance.py train
```

Run this again whenever you add a new person.

### 3. Start taking attendance

```bash
python smart_attendance.py run
```

A video window opens. Recognized people get a green box and are marked present. Unrecognized faces show as "Unknown" in red. Press `q` to stop.

### 4. View a report

```bash
python smart_attendance.py report                  # today
python smart_attendance.py report --date 2026-10-18
```

### Options

| Flag | Applies to | Description |
|------|------------|-------------|
| `--camera N` | all | Camera index, default 0 (place before the command) |
| `--id` | register | Unique numeric ID |
| `--name` | register | Person's name |
| `--samples` | register | Number of face samples, default 30 |
| `--date` | report | Date in YYYY-MM-DD format |

Example with a second camera: `python smart_attendance.py --camera 1 run`

## Output

Attendance is saved to `attendance/YYYY-MM-DD.csv`:

```csv
id,name,date,time
1,Asha,2026-10-18,09:02:14
2,Ravi,2026-10-18,09:05:41
```

## Project structure

```
smart-attendance/
├── smart_attendance.py   # all commands
├── README.md
├── dataset/              # created on register (face images)
├── model/                # created on train (model + names)
└── attendance/           # created on run (daily CSV files)
```

## Tuning

At the top of `smart_attendance.py`:

- `CONFIDENCE_THRESHOLD` (default 70): lower values are stricter. If strangers get recognized, lower it. If registered people show as Unknown, raise it.
- `CONFIRM_FRAMES` (default 5): how many consecutive frames must match before attendance is marked.

For best accuracy, register people in lighting similar to where the system will run, and capture samples from a few angles.

## Privacy and limitations

- Face images and names are stored locally in `dataset/` and `model/`. Get consent from everyone you register, and follow the privacy laws that apply to you.
- Do not commit `dataset/`, `model/` or `attendance/` to a public repository. Add them to `.gitignore`:

  ```
  dataset/
  model/
  attendance/
  ```
- LBPH is lightweight but less accurate than deep-learning methods. It can be fooled by a printed photo, so it is not suitable for high-security use without liveness detection.

## Ideas for contributions

- Liveness or blink detection to block photo spoofing
- Deep-learning recognition (for example `face_recognition` or ArcFace)
- Excel export and email reports
- Web dashboard with Flask or Django
- Late-arrival cutoff and check-out time

## License

MIT. Add a `LICENSE` file to your repository if you want to publish under this license.

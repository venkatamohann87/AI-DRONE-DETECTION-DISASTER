

import os
import cv2
import numpy as np
import pandas as pd

from collections import defaultdict, deque
from pathlib import Path

from IPython.display import display, clear_output
from PIL import Image

# --------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------

try:
    import geojson
except ImportError:
    geojson = None

try:
    import folium
except ImportError:
    folium = None

try:
    from geopy.distance import geodesic
except ImportError:
    geodesic = None

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics is not installed.\n"
        "Run in Jupyter:\n"
        "%pip install ultralytics"
    )

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

# RGB VIDEO
RGB_VIDEO_PATH = (
    r"t1.webm"
)

# Optional thermal video.
# Use None if you do not have thermal video.
THERMAL_VIDEO_PATH = "flight_clip_thermal.mp4"

# Optional thermal homography.
THERMAL_HOMOGRAPHY_PATH = "thermal_homography.npy"

# Optional sonar CSV.
SONAR_CSV = "sonar_detections.csv"

SONAR_MATCH_RADIUS_METERS = 15.0
SONAR_TIME_WINDOW_SECONDS = 5.0

# Optional UWB (Ultra-Wideband) CSV for close-range survivor
# confirmation. UWB has no useful range/GPS accuracy, so it is
# matched on time only (not location) - it confirms "something
# is very close to the drone right now", not "where".
UWB_CSV = "uwb_detections.csv"
UWB_TIME_WINDOW_SECONDS = 3.0

# Telemetry CSV.
# IMPORTANT: the CSV contents must NOT be pasted into this .py file.
#
# telemetry.csv example:
# timestamp,lat,lon,altitude_m,yaw_deg
# 2026-09-10 16:32:16,13.0827,80.2707,80,20
#
TELEMETRY_CSV = "telemetry.csv"

# YOLO model
MODEL_WEIGHTS = "yolov8n.pt"

# Optional P2 model
P2_MODEL_WEIGHTS = "best_p2.pt"

# Output files
OUTPUT_VIDEO = "survivor_detection_output_1080p.mp4"
OUTPUT_GEOJSON = "triage_output.geojson"
OUTPUT_MAP_HTML = "triage_map.html"

# Live survivor results CSV - updated after every processed frame.
LIVE_RESULTS_CSV = "survivor_detection_result.csv"

# Evidence images
EVIDENCE_DIR = "survivor_evidence"

# Output resolution
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

# Camera
HORIZONTAL_FOV_DEG = 84.0

# Detection
CONF_THRESHOLD = 0.25

# Tracking
MIN_CONFIRMATION_FRAMES = 5
MAX_TRACK_MISSING_FRAMES = 30
MAX_TRACK_DISTANCE_PIXELS = 120

# Geographic duplicate removal
DEDUP_DISTANCE_METERS = 4.0

# SAHI
USE_SAHI = True
SAHI_SLICE_HEIGHT = 640
SAHI_SLICE_WIDTH = 640
SAHI_OVERLAP_HEIGHT = 0.20
SAHI_OVERLAP_WIDTH = 0.20

# Temporal processing
TEMPORAL_WINDOW = 3

# Environment
# normal / dust / heavy_dust / smoke / fog / rain /
# low_light / motion_blur
ENVIRONMENT = "heavy_dust"

# Jupyter preview
# 1 = every frame, 5 = every fifth frame, etc.
DISPLAY_EVERY = 5
PREVIEW_WIDTH = 1280


# ==============================================================
# 2. FILE CHECKS
# ==============================================================

if not os.path.isfile(RGB_VIDEO_PATH):
    raise FileNotFoundError(
        "\nRGB video not found:\n"
        f"{RGB_VIDEO_PATH}\n\n"
        "Change RGB_VIDEO_PATH in the configuration section."
    )

USE_THERMAL = (
    THERMAL_VIDEO_PATH is not None
    and os.path.isfile(THERMAL_VIDEO_PATH)
)

USE_TELEMETRY = os.path.isfile(TELEMETRY_CSV)
USE_SONAR = os.path.isfile(SONAR_CSV)
USE_UWB = os.path.isfile(UWB_CSV)
USE_THERMAL_CALIBRATION = os.path.isfile(
    THERMAL_HOMOGRAPHY_PATH
)

Path(EVIDENCE_DIR).mkdir(
    parents=True,
    exist_ok=True
)

if not USE_THERMAL:
    print("WARNING: Thermal video not found. Thermal fusion is OFF.")

if not USE_TELEMETRY:
    print(
        "WARNING: telemetry.csv not found. "
        "GPS localization and map output are OFF."
    )

if not USE_SONAR:
    print("INFO: sonar_detections.csv not found. Sonar is OFF.")

if not USE_UWB:
    print("INFO: uwb_detections.csv not found. UWB is OFF.")


# ==============================================================
# 3. LOAD MODELS
# ==============================================================

print("\n==============================================")
print("      ADVANCED SURVIVOR DETECTION SYSTEM")
print("==============================================")

print("\nLoading YOLO model...")
model = YOLO(MODEL_WEIGHTS)
print("YOLO loaded successfully.")

p2_model = None
P2_MODEL_AVAILABLE = False

if os.path.isfile(P2_MODEL_WEIGHTS):
    print(f"P2 model found: {P2_MODEL_WEIGHTS}")
    p2_model = YOLO(P2_MODEL_WEIGHTS)
    P2_MODEL_AVAILABLE = True
else:
    print("P2 model not found. Normal YOLO/SAHI will be used.")

print(f"Thermal video : {USE_THERMAL}")
print(f"Thermal calib : {USE_THERMAL_CALIBRATION}")
print(f"Telemetry     : {USE_TELEMETRY}")
print(f"Sonar         : {USE_SONAR}")
print(f"UWB           : {USE_UWB}")
print(f"SAHI          : {USE_SAHI and SAHI_AVAILABLE}")
print(f"P2 model      : {P2_MODEL_AVAILABLE}")


# ==============================================================
# 4. ENVIRONMENTAL ENHANCEMENT
# ==============================================================

def clahe_enhancement(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.merge((l, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def denoise_frame(frame):
    return cv2.bilateralFilter(
        frame,
        5,
        50,
        50
    )


def sharpen_frame(frame):
    blurred = cv2.GaussianBlur(
        frame,
        (0, 0),
        2
    )

    return cv2.addWeighted(
        frame,
        1.3,
        blurred,
        -0.3,
        0
    )


def dehaze_simple(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.merge((l, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def remove_rain_noise(frame):
    return cv2.bilateralFilter(
        frame,
        5,
        60,
        60
    )


def enhance_frame(frame, condition):
    if condition == "normal":
        return frame

    if condition == "dust":
        return clahe_enhancement(frame)

    if condition == "heavy_dust":
        frame = denoise_frame(frame)
        frame = clahe_enhancement(frame)
        frame = dehaze_simple(frame)
        frame = sharpen_frame(frame)
        return frame

    if condition == "smoke":
        frame = dehaze_simple(frame)
        return clahe_enhancement(frame)

    if condition == "fog":
        frame = dehaze_simple(frame)
        return clahe_enhancement(frame)

    if condition == "rain":
        frame = remove_rain_noise(frame)
        return clahe_enhancement(frame)

    if condition == "low_light":
        return clahe_enhancement(frame)

    if condition == "motion_blur":
        frame = denoise_frame(frame)
        return sharpen_frame(frame)

    return frame


# ==============================================================
# 5. THERMAL FUSION
# ==============================================================

def load_thermal_homography():
    if not USE_THERMAL_CALIBRATION:
        return None

    try:
        H = np.load(THERMAL_HOMOGRAPHY_PATH)
        H = np.asarray(H, dtype=np.float64)

        if H.shape != (3, 3):
            raise ValueError(
                "Thermal homography must be a 3x3 matrix."
            )

        return H

    except Exception as error:
        print(
            f"WARNING: thermal calibration could not be loaded: "
            f"{error}"
        )
        return None


THERMAL_HOMOGRAPHY = load_thermal_homography()


def prepare_thermal_gray(
    thermal_frame,
    target_width,
    target_height
):
    if len(thermal_frame.shape) == 3:
        gray = cv2.cvtColor(
            thermal_frame,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = thermal_frame.copy()

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    if THERMAL_HOMOGRAPHY is not None:
        registered = cv2.warpPerspective(
            gray,
            THERMAL_HOMOGRAPHY,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR
        )
    else:
        registered = cv2.resize(
            gray,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

    return registered


def fuse_rgb_thermal(
    rgb_frame,
    thermal_frame
):
    h, w = rgb_frame.shape[:2]

    thermal_gray = prepare_thermal_gray(
        thermal_frame,
        w,
        h
    )

    thermal_colored = cv2.applyColorMap(
        thermal_gray,
        cv2.COLORMAP_JET
    )

    fused = cv2.addWeighted(
        rgb_frame,
        0.72,
        thermal_colored,
        0.28,
        0
    )

    return fused, thermal_gray


def thermal_box_score(
    thermal_gray,
    x1,
    y1,
    x2,
    y2
):
    if thermal_gray is None:
        return 0.0

    h, w = thermal_gray.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = thermal_gray[y1:y2, x1:x2]

    if roi.size == 0:
        return 0.0

    value = float(
        np.percentile(roi, 90)
    ) / 255.0

    return float(
        np.clip(value, 0.0, 1.0)
    )


# ==============================================================
# 6. TEMPORAL PROCESSING
# ==============================================================

previous_frames = deque(
    maxlen=TEMPORAL_WINDOW
)


def temporal_enhancement(current_frame):
    previous_frames.append(
        current_frame.copy()
    )

    if len(previous_frames) < 2:
        return current_frame

    average = np.mean(
        list(previous_frames),
        axis=0
    ).astype(np.uint8)

    return cv2.addWeighted(
        current_frame,
        0.70,
        average,
        0.30,
        0
    )


# ==============================================================
# 7. TELEMETRY
# ==============================================================

def load_telemetry(csv_path):
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"]
    )

    required = [
        "timestamp",
        "lat",
        "lon",
        "altitude_m",
        "yaw_deg"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Telemetry CSV is missing columns: "
            + ", ".join(missing)
        )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def get_telemetry_for_frame(
    telemetry_df,
    frame_idx,
    fps
):
    if telemetry_df is None or telemetry_df.empty:
        return None

    elapsed_seconds = frame_idx / fps

    start_time = telemetry_df[
        "timestamp"
    ].iloc[0]

    target_time = (
        start_time
        + pd.Timedelta(
            seconds=elapsed_seconds
        )
    )

    idx = (
        telemetry_df["timestamp"]
        - target_time
    ).abs().idxmin()

    return telemetry_df.loc[idx]


# ==============================================================
# 8. SONAR
# ==============================================================

sonar_df = None


def load_sonar(csv_path):
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"]
    )

    required = [
        "timestamp",
        "lat",
        "lon",
        "depth_m",
        "confidence",
        "target_type"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Sonar CSV is missing columns: "
            + ", ".join(missing)
        )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def match_sonar(
    lat,
    lon,
    frame_time,
    sonar_data
):
    if (
        sonar_data is None
        or sonar_data.empty
        or geodesic is None
    ):
        return False, 0.0, ""

    time_delta = (
        sonar_data["timestamp"]
        - frame_time
    ).abs()

    candidates = sonar_data[
        time_delta
        <= pd.Timedelta(
            seconds=SONAR_TIME_WINDOW_SECONDS
        )
    ]

    if candidates.empty:
        return False, 0.0, ""

    best_conf = 0.0
    best_type = ""

    for _, row in candidates.iterrows():
        distance = geodesic(
            (lat, lon),
            (
                float(row["lat"]),
                float(row["lon"])
            )
        ).meters

        if distance <= SONAR_MATCH_RADIUS_METERS:
            confidence = float(
                np.clip(
                    row["confidence"],
                    0.0,
                    1.0
                )
            )

            if confidence > best_conf:
                best_conf = confidence
                best_type = str(
                    row["target_type"]
                )

    return best_conf > 0.0, best_conf, best_type


# ==============================================================
# 8B. UWB (CLOSE-RANGE SURVIVOR CONFIRMATION)
# ==============================================================

uwb_df = None


def load_uwb(csv_path):
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"]
    )

    required = [
        "timestamp",
        "distance_m",
        "confidence"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "UWB CSV is missing columns: "
            + ", ".join(missing)
        )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def match_uwb(
    frame_time,
    uwb_data
):
    if (
        uwb_data is None
        or uwb_data.empty
    ):
        return 0.0

    time_delta = (
        uwb_data["timestamp"]
        - frame_time
    ).abs()

    candidates = uwb_data[
        time_delta
        <= pd.Timedelta(
            seconds=UWB_TIME_WINDOW_SECONDS
        )
    ]

    if candidates.empty:
        return 0.0

    best_conf = float(
        candidates["confidence"].max()
    )

    return float(
        np.clip(best_conf, 0.0, 1.0)
    )


# ==============================================================
# 9. PIXEL TO GPS
# ==============================================================

def pixel_to_latlon(
    px,
    py,
    telemetry_row,
    image_width,
    image_height,
    h_fov_deg
):
    lat = float(telemetry_row["lat"])
    lon = float(telemetry_row["lon"])

    altitude = float(
        telemetry_row["altitude_m"]
    )

    yaw_deg = float(
        telemetry_row["yaw_deg"]
    )

    h_fov_rad = np.radians(
        h_fov_deg
    )

    v_fov_rad = (
        h_fov_rad
        * image_height
        / image_width
    )

    dx_norm = (
        px - image_width / 2
    ) / (image_width / 2)

    dy_norm = (
        py - image_height / 2
    ) / (image_height / 2)

    ground_dx = (
        altitude
        * np.tan(
            dx_norm
            * h_fov_rad
            / 2
        )
    )

    ground_dy = (
        altitude
        * np.tan(
            dy_norm
            * v_fov_rad
            / 2
        )
    )

    yaw_rad = np.radians(
        yaw_deg
    )

    north_offset = (
        ground_dx * np.sin(yaw_rad)
        + ground_dy * np.cos(yaw_rad)
    )

    east_offset = (
        ground_dx * np.cos(yaw_rad)
        - ground_dy * np.sin(yaw_rad)
    )

    meters_per_deg_lat = 111320.0

    meters_per_deg_lon = (
        111320.0
        * np.cos(np.radians(lat))
    )

    meters_per_deg_lon = max(
        abs(meters_per_deg_lon),
        1e-6
    )

    new_lat = (
        lat
        + north_offset
        / meters_per_deg_lat
    )

    new_lon = (
        lon
        + east_offset
        / meters_per_deg_lon
    )

    return new_lat, new_lon


# ==============================================================
# 10. NORMAL YOLO DETECTION
# ==============================================================

def normal_detection(
    detector,
    frame
):
    results = detector.predict(
        frame,
        conf=CONF_THRESHOLD,
        classes=[0],
        verbose=False
    )

    detections = []

    if (
        results
        and results[0].boxes is not None
        and len(results[0].boxes) > 0
    ):
        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        confs = (
            results[0]
            .boxes
            .conf
            .cpu()
            .numpy()
        )

        classes = (
            results[0]
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, conf, cls in zip(
            boxes,
            confs,
            classes
        ):
            if cls != 0:
                continue

            x1, y1, x2, y2 = box

            detections.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                    int(cls)
                ]
            )

    return detections


# ==============================================================
# 11. SAHI
# ==============================================================

_sahi_detection_model = None


def get_sahi_model():
    global _sahi_detection_model

    if _sahi_detection_model is not None:
        return _sahi_detection_model

    if not SAHI_AVAILABLE:
        return None

    try:
        _sahi_detection_model = (
            AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=MODEL_WEIGHTS,
                confidence_threshold=CONF_THRESHOLD,
                device="cuda:0"
            )
        )
    except Exception:
        try:
            _sahi_detection_model = (
                AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=MODEL_WEIGHTS,
                    confidence_threshold=CONF_THRESHOLD,
                    device="cpu"
                )
            )
        except Exception as error:
            print(
                f"WARNING: SAHI initialization failed: {error}"
            )
            _sahi_detection_model = None

    return _sahi_detection_model


def sahi_detection(frame):
    detection_model = get_sahi_model()

    if detection_model is None:
        return normal_detection(
            model,
            frame
        )

    try:
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=SAHI_SLICE_HEIGHT,
            slice_width=SAHI_SLICE_WIDTH,
            overlap_height_ratio=SAHI_OVERLAP_HEIGHT,
            overlap_width_ratio=SAHI_OVERLAP_WIDTH
        )
    except Exception as error:
        print(
            f"WARNING: SAHI inference failed: {error}"
        )
        return normal_detection(
            model,
            frame
        )

    detections = []

    for obj in result.object_prediction_list:
        category_id = int(
            obj.category.id
        )

        if category_id != 0:
            continue

        bbox = obj.bbox

        detections.append(
            [
                float(bbox.minx),
                float(bbox.miny),
                float(bbox.maxx),
                float(bbox.maxy),
                float(obj.score.value),
                category_id
            ]
        )

    return detections


# ==============================================================
# 12. IOU + LIGHTWEIGHT TRACKER
# ==============================================================

def calculate_iou(
    box_a,
    box_b
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_detections(
    pred_boxes,
    gt_boxes,
    iou_thresh=0.5
):
    """
    Optional accuracy check against hand-labeled ground truth
    boxes for a frame or clip. Not wired into the main pipeline
    automatically - call it yourself with your own gt_boxes list
    (e.g. [[x1, y1, x2, y2], ...]) if you want a recall/false
    positive readout for a validation clip.
    """
    true_positives = 0
    false_positives = 0

    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue

            iou = calculate_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= iou_thresh:
            true_positives += 1
            matched_gt.add(best_gt_index)
        else:
            false_positives += 1

    recall = true_positives / max(len(gt_boxes), 1)

    return {
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives
    }


def associate_detections(
    detections,
    active_tracks,
    next_track_id
):
    assignments = []
    used_track_ids = set()

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        best_id = None
        best_score = float("inf")

        for tid, track in active_tracks.items():
            if tid in used_track_ids:
                continue

            distance = np.sqrt(
                (cx - track["cx"]) ** 2
                + (cy - track["cy"]) ** 2
            )

            iou = calculate_iou(
                (x1, y1, x2, y2),
                track["box"]
            )

            if (
                distance <= MAX_TRACK_DISTANCE_PIXELS
                or iou >= 0.10
            ):
                score = (
                    distance
                    - iou * MAX_TRACK_DISTANCE_PIXELS
                )

                if score < best_score:
                    best_score = score
                    best_id = tid

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1

        used_track_ids.add(best_id)

        assignments.append(
            (
                best_id,
                detection
            )
        )

    return assignments, next_track_id


def create_track_record():
    return {
        "positions": [],
        "confidences": [],
        "best_conf": 0.0,
        "best_frame": None,
        "frames_seen": 0,
        "class_name": "person",
        "source": "RGB",
        "confirmed": False,
        "last_seen_frame": -1,
        "moving": False,
        "thermal": False,
        "thermal_score": 0.0,
        "sonar_score": 0.0,
        "sonar_target_type": "",
        "uwb_score": 0.0,
        "priority_score": 0.0,
        "priority_level": "LOW"
    }


# ==============================================================
# 13. PRIORITY SCORING
# ==============================================================

PRIORITY_CRITICAL = 85
PRIORITY_HIGH = 70
PRIORITY_MEDIUM = 50

PRIORITY_W_CONFIDENCE = 0.40
PRIORITY_W_PERSISTENCE = 0.25
PRIORITY_W_MOTION = 0.10
PRIORITY_W_THERMAL = 0.10
PRIORITY_W_SONAR = 0.10
PRIORITY_W_UWB = 0.05


def is_moving(
    positions,
    threshold_m=2.0
):
    if len(positions) < 2:
        return False

    if geodesic is None:
        return False

    distance = geodesic(
        positions[0],
        positions[-1]
    ).meters

    return distance > threshold_m


def calculate_priority_score(
    confidence,
    frames_seen,
    positions,
    thermal_score=0.0,
    sonar_score=0.0,
    uwb_score=0.0
):
    confidence_score = (
        float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )
        * 100
    )

    persistence_score = (
        min(
            frames_seen
            / max(
                MIN_CONFIRMATION_FRAMES * 2,
                1
            ),
            1.0
        )
        * 100
    )

    motion_score = (
        100.0
        if is_moving(positions)
        else 0.0
    )

    thermal_score = (
        float(
            np.clip(
                thermal_score,
                0.0,
                1.0
            )
        )
        * 100
    )

    sonar_score = (
        float(
            np.clip(
                sonar_score,
                0.0,
                1.0
            )
        )
        * 100
    )

    uwb_score = (
        float(
            np.clip(
                uwb_score,
                0.0,
                1.0
            )
        )
        * 100
    )

    score = (
        PRIORITY_W_CONFIDENCE * confidence_score
        + PRIORITY_W_PERSISTENCE * persistence_score
        + PRIORITY_W_MOTION * motion_score
        + PRIORITY_W_THERMAL * thermal_score
        + PRIORITY_W_SONAR * sonar_score
        + PRIORITY_W_UWB * uwb_score
    )

    return float(
        np.clip(score, 0, 100)
    )


def priority_level(score):
    if score >= PRIORITY_CRITICAL:
        return "CRITICAL"

    if score >= PRIORITY_HIGH:
        return "HIGH"

    if score >= PRIORITY_MEDIUM:
        return "MEDIUM"

    return "LOW"


# ==============================================================
# 14. LABEL + DASHBOARD
# ==============================================================

def draw_label(
    frame,
    text,
    x1,
    y1,
    box_color
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    label_x1 = int(x1)
    label_y2 = int(y1)

    label_x2 = (
        label_x1
        + text_width
        + 12
    )

    label_y1 = (
        label_y2
        - text_height
        - baseline
        - 10
    )

    if label_y1 < 0:
        label_y1 = label_y2 + 2
        label_y2 = (
            label_y1
            + text_height
            + baseline
            + 10
        )

    if label_x2 >= frame.shape[1]:
        shift = (
            label_x2
            - frame.shape[1]
            + 2
        )
        label_x1 -= shift
        label_x2 -= shift

    cv2.rectangle(
        frame,
        (label_x1, label_y1),
        (label_x2, label_y2),
        box_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (
            label_x1 + 6,
            label_y2 - 7
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def draw_dashboard(
    frame,
    confirmed_count,
    possible_count,
    total_unique,
    environment,
    detector_mode,
    thermal_enabled,
    gps_enabled,
    sonar_enabled,
    critical_count,
    uwb_enabled=False
):
    panel_x1 = 20
    panel_y1 = 20
    panel_x2 = 650
    panel_y2 = 285

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (255, 255, 255),
        2
    )

    lines = [
        "DISASTER SURVIVOR DETECTION",
        f"CONFIRMED SURVIVORS : {confirmed_count}",
        f"POSSIBLE SURVIVORS  : {possible_count}",
        f"UNIQUE TRACKS       : {total_unique}",
        f"ENVIRONMENT         : {environment}",
        f"DETECTOR            : {detector_mode}",
        f"THERMAL             : {'ON' if thermal_enabled else 'OFF'}",
        f"GPS                 : {'ON' if gps_enabled else 'OFF'}",
        f"SONAR               : {'ON' if sonar_enabled else 'OFF'}",
        f"UWB                 : {'ON' if uwb_enabled else 'OFF'}",
        f"CRITICAL PRIORITY   : {critical_count}"
    ]

    y = 50

    for index, text in enumerate(lines):
        scale = 0.68 if index == 0 else 0.55

        cv2.putText(
            frame,
            text,
            (35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 25 if index == 0 else 24


# ==============================================================
# 15. EVIDENCE CROP
# ==============================================================

def save_evidence_crop(
    rgb_frame,
    x1,
    y1,
    x2,
    y2,
    track_id,
    frame_idx,
    confidence
):
    h, w = rgb_frame.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = rgb_frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    path = os.path.join(
        EVIDENCE_DIR,
        f"evidence_track_{track_id}_"
        f"frame_{frame_idx}_"
        f"conf_{confidence:.2f}.jpg"
    )

    cv2.imwrite(
        path,
        crop
    )

    return path


# ==============================================================
# 16. GEOGRAPHIC DEDUPLICATION
# ==============================================================

def dedup_by_distance(
    track_records,
    max_distance_m=DEDUP_DISTANCE_METERS
):
    if geodesic is None:
        print(
            "geopy is not installed; "
            "GPS deduplication skipped."
        )
        return dict(track_records)

    merged = {}
    used = set()

    track_ids = list(
        track_records.keys()
    )

    averages = {}

    for tid in track_ids:
        positions = (
            track_records[tid]["positions"]
        )

        if not positions:
            continue

        averages[tid] = (
            float(
                np.mean(
                    [p[0] for p in positions]
                )
            ),
            float(
                np.mean(
                    [p[1] for p in positions]
                )
            )
        )

    for i, tid_a in enumerate(track_ids):
        if tid_a in used:
            continue

        if tid_a not in averages:
            continue

        group = [tid_a]

        for tid_b in track_ids[i + 1:]:
            if tid_b in used:
                continue

            if tid_b not in averages:
                continue

            distance_m = geodesic(
                averages[tid_a],
                averages[tid_b]
            ).meters

            if distance_m <= max_distance_m:
                group.append(tid_b)
                used.add(tid_b)

        best_tid = max(
            group,
            key=lambda t:
            track_records[t]["best_conf"]
        )

        merged_record = dict(
            track_records[best_tid]
        )

        merged_record["merged_from"] = list(
            group
        )

        merged[best_tid] = merged_record

        used.add(tid_a)

    return merged


# ==============================================================
# 17. GEOJSON + MAP
# ==============================================================

def build_triage_output(
    track_records
):
    if geojson is None:
        raise ImportError(
            "geojson is not installed.\n"
            "Run:\n"
            "%pip install geojson"
        )

    features = []

    for tid, record in track_records.items():
        if not record["positions"]:
            continue

        lats = [
            p[0]
            for p in record["positions"]
        ]

        lons = [
            p[1]
            for p in record["positions"]
        ]

        avg_lat = float(
            np.mean(lats)
        )

        avg_lon = float(
            np.mean(lons)
        )

        feature = geojson.Feature(
            geometry=geojson.Point(
                (
                    avg_lon,
                    avg_lat
                )
            ),
            properties={
                "track_id": int(tid),
                "class": record["class_name"],
                "confidence": float(
                    record["best_conf"]
                ),
                "confirmed": bool(
                    record["confirmed"]
                ),
                "frames_seen": int(
                    record["frames_seen"]
                ),
                "moving": bool(
                    is_moving(
                        record["positions"]
                    )
                ),
                "priority_score": float(
                    record["priority_score"]
                ),
                "priority_level": record[
                    "priority_level"
                ],
                "thermal_score": float(
                    record["thermal_score"]
                ),
                "sonar_score": float(
                    record["sonar_score"]
                ),
                "sonar_target_type": record[
                    "sonar_target_type"
                ],
                "uwb_score": float(
                    record["uwb_score"]
                ),
                "estimated_count": 1,
                "merged_from": record.get(
                    "merged_from",
                    [tid]
                ),
                "evidence_thumbnail": record[
                    "best_frame"
                ],
                "source": record["source"],
                "thermal": bool(
                    record["thermal"]
                )
            }
        )

        features.append(feature)

    features.sort(
        key=lambda f:
        f["properties"]["confidence"],
        reverse=True
    )

    return geojson.FeatureCollection(
        features
    )


# ==============================================================
# OFFLINE QUEUE
#
# If writing the GeoJSON output fails (e.g. no write access to a
# mounted/network drive, or the process is interrupted), the
# results are queued locally as plain JSON so nothing is lost.
# ==============================================================

OFFLINE_QUEUE_FILE = "offline_queue.json"


def save_offline_queue(data):
    import json

    with open(
        OFFLINE_QUEUE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(data, f)

    print(
        f"Queued results locally: {OFFLINE_QUEUE_FILE}"
    )


def load_offline_queue():
    import json

    if not os.path.exists(OFFLINE_QUEUE_FILE):
        return []

    with open(
        OFFLINE_QUEUE_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_geojson(
    feature_collection,
    path
):
    try:
        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            geojson.dump(
                feature_collection,
                f,
                indent=2
            )

        print(
            f"GeoJSON saved: {path}"
        )

    except OSError as error:
        print(
            f"WARNING: could not write GeoJSON ({error}). "
            "Saving to the offline queue instead."
        )
        save_offline_queue(feature_collection)


def render_map(
    feature_collection,
    output_html
):
    if folium is None:
        print(
            "folium is not installed; map skipped."
        )
        return

    if not feature_collection["features"]:
        print(
            "No GPS detections to map."
        )
        return

    first_lon, first_lat = (
        feature_collection[
            "features"
        ][0]["geometry"]["coordinates"]
    )

    m = folium.Map(
        location=[
            first_lat,
            first_lon
        ],
        zoom_start=17
    )

    for feature in feature_collection[
        "features"
    ]:
        lon, lat = feature[
            "geometry"
        ]["coordinates"]

        props = feature[
            "properties"
        ]

        marker_color = (
            "red"
            if props["confirmed"]
            else "orange"
        )

        popup_text = (
            f"Track ID: {props['track_id']}"
            f"<br>Confidence: "
            f"{props['confidence']:.2f}"
            f"<br>Confirmed: "
            f"{props['confirmed']}"
            f"<br>Frames: "
            f"{props['frames_seen']}"
            f"<br>Priority: "
            f"{props['priority_level']} "
            f"({props['priority_score']:.0f}/100)"
            f"<br>Thermal: "
            f"{props['thermal_score']:.2f}"
            f"<br>Sonar: "
            f"{props['sonar_score']:.2f}"
            f"<br>UWB: "
            f"{props['uwb_score']:.2f}"
        )

        folium.CircleMarker(
            location=[
                lat,
                lon
            ],
            radius=8,
            color=marker_color,
            fill=True,
            fill_opacity=0.8,
            popup=popup_text
        ).add_to(m)

    m.save(
        output_html
    )

    print(
        f"Map saved: {output_html}"
    )


# ==============================================================
# 18. FIT FRAME TO 1920x1080
# ==============================================================

def fit_to_1080p(frame):
    target_w = OUTPUT_WIDTH
    target_h = OUTPUT_HEIGHT

    h, w = frame.shape[:2]

    if w <= 0 or h <= 0:
        return np.zeros(
            (
                target_h,
                target_w,
                3
            ),
            dtype=np.uint8
        )

    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (
            target_h,
            target_w,
            3
        ),
        dtype=np.uint8
    )

    x_offset = (
        target_w - new_w
    ) // 2

    y_offset = (
        target_h - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# ==============================================================
# LIVE SURVIVOR RESULTS CSV
# ==============================================================

def update_live_survivor_csv(track_records):
    """
    Rewrite survivor_detection_result.csv with the latest
    survivor information after every processed video frame.
    """

    survivor_rows = []

    for tid, record in track_records.items():

        if not record.get("positions"):
            continue

        latitude = float(
            np.mean(
                [p[0] for p in record["positions"]]
            )
        )

        longitude = float(
            np.mean(
                [p[1] for p in record["positions"]]
            )
        )

        survivor_rows.append({
            "ID": int(tid),
            "Confidence": round(
                float(record.get("best_conf", 0.0)),
                2
            ),
            "Latitude": round(latitude, 6),
            "Longitude": round(longitude, 6),
            "Priority": record.get(
                "priority_level",
                "LOW"
            ),
            "Priority Score": round(
                float(
                    record.get(
                        "priority_score",
                        0.0
                    )
                ),
                1
            ),
            "Confirmed": bool(
                record.get(
                    "confirmed",
                    False
                )
            ),
            "Frames": int(
                record.get(
                    "frames_seen",
                    0
                )
            )
        })

    survivor_rows.sort(
        key=lambda x: x["Confidence"],
        reverse=True
    )

    live_df = pd.DataFrame(
        survivor_rows,
        columns=[
            "ID",
            "Confidence",
            "Latitude",
            "Longitude",
            "Priority",
            "Priority Score",
            "Confirmed",
            "Frames"
        ]
    )

    # Write the complete current state after this frame.
    live_df.to_csv(
        LIVE_RESULTS_CSV,
        index=False
    )

    return live_df


# ==============================================================
# 19. JUPYTER FRAME DISPLAY
# ==============================================================

def display_frame_jupyter(
    frame,
    frame_idx,
    total_frames
):
    display_frame = frame.copy()

    h, w = display_frame.shape[:2]

    if w > PREVIEW_WIDTH:
        scale = (
            PREVIEW_WIDTH / w
        )

        display_frame = cv2.resize(
            display_frame,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    display_rgb = cv2.cvtColor(
        display_frame,
        cv2.COLOR_BGR2RGB
    )

    clear_output(
        wait=True
    )

    display(
        Image.fromarray(
            display_rgb
        )
    )

    print(
        f"Frame {frame_idx}/{total_frames}"
    )


# ==============================================================
# 20. MAIN VIDEO PROCESSING
# ==============================================================

def process_recorded_video():
    global previous_frames

    previous_frames.clear()

    rgb_cap = cv2.VideoCapture(
        RGB_VIDEO_PATH
    )

    if not rgb_cap.isOpened():
        raise RuntimeError(
            "Could not open RGB video."
        )

    thermal_cap = None

    if USE_THERMAL:
        thermal_cap = cv2.VideoCapture(
            THERMAL_VIDEO_PATH
        )

        if not thermal_cap.isOpened():
            print(
                "WARNING: thermal video could not be opened."
            )
            thermal_cap.release()
            thermal_cap = None

    fps = (
        rgb_cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30
    )

    width = int(
        rgb_cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        rgb_cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        rgb_cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("\nVideo information")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps}")
    print(f"Frames     : {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    output = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT
        )
    )

    if not output.isOpened():
        rgb_cap.release()

        if thermal_cap is not None:
            thermal_cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    track_records = defaultdict(
        create_track_record
    )

    active_tracks = {}
    next_track_id = 1
    frame_idx = 0

    try:
        while True:
            ret_rgb, rgb_frame = (
                rgb_cap.read()
            )

            if not ret_rgb:
                break

            input_frame = rgb_frame.copy()
            thermal_gray_current = None
            thermal_used_this_frame = False

            # --------------------------------------------------
            # Thermal
            # --------------------------------------------------

            if thermal_cap is not None:
                ret_thermal, thermal_frame = (
                    thermal_cap.read()
                )

                if ret_thermal:
                    (
                        input_frame,
                        thermal_gray_current
                    ) = fuse_rgb_thermal(
                        rgb_frame,
                        thermal_frame
                    )

                    thermal_used_this_frame = True

            # --------------------------------------------------
            # Environment
            # --------------------------------------------------

            input_frame = enhance_frame(
                input_frame,
                ENVIRONMENT
            )

            # --------------------------------------------------
            # Temporal
            # --------------------------------------------------

            if ENVIRONMENT == "motion_blur":
                input_frame = temporal_enhancement(
                    input_frame
                )

            # --------------------------------------------------
            # Detection
            # --------------------------------------------------

            if P2_MODEL_AVAILABLE:
                detections = normal_detection(
                    p2_model,
                    input_frame
                )

                detection_mode = "P2"

                (
                    assignments,
                    next_track_id
                ) = associate_detections(
                    detections,
                    active_tracks,
                    next_track_id
                )

            elif USE_SAHI and SAHI_AVAILABLE:
                detections = sahi_detection(
                    input_frame
                )

                detection_mode = "SAHI"

                (
                    assignments,
                    next_track_id
                ) = associate_detections(
                    detections,
                    active_tracks,
                    next_track_id
                )

            else:
                # ByteTrack through Ultralytics.
                results = model.track(
                    input_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=CONF_THRESHOLD,
                    classes=[0],
                    verbose=False
                )

                assignments = []

                detection_mode = "ByteTrack"

                if results:
                    result = results[0]

                    if (
                        result.boxes is not None
                        and len(result.boxes) > 0
                    ):
                        boxes = (
                            result.boxes.xyxy
                            .cpu()
                            .numpy()
                        )

                        confs = (
                            result.boxes.conf
                            .cpu()
                            .numpy()
                        )

                        if result.boxes.id is not None:
                            ids = (
                                result.boxes.id
                                .cpu()
                                .numpy()
                                .astype(int)
                            )
                        else:
                            ids = np.arange(
                                1,
                                len(boxes) + 1
                            )

                        for box, conf, tid in zip(
                            boxes,
                            confs,
                            ids
                        ):
                            x1, y1, x2, y2 = box

                            assignments.append(
                                (
                                    int(tid),
                                    [
                                        float(x1),
                                        float(y1),
                                        float(x2),
                                        float(y2),
                                        float(conf),
                                        0
                                    ]
                                )
                            )

            # --------------------------------------------------
            # Process detections
            # --------------------------------------------------

            for track_id, detection in assignments:
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    cls
                ) = detection

                x1 = max(
                    0,
                    min(width - 1, int(x1))
                )

                y1 = max(
                    0,
                    min(height - 1, int(y1))
                )

                x2 = max(
                    0,
                    min(width, int(x2))
                )

                y2 = max(
                    0,
                    min(height, int(y2))
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (
                    x1 + x2
                ) / 2

                cy = (
                    y1 + y2
                ) / 2

                track_id = int(
                    track_id
                )

                if track_id not in track_records:
                    track_records[
                        track_id
                    ] = create_track_record()

                record = track_records[
                    track_id
                ]

                active_tracks[
                    track_id
                ] = {
                    "cx": cx,
                    "cy": cy,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "frame": frame_idx
                }

                record["frames_seen"] += 1
                record["last_seen_frame"] = frame_idx
                record["confidences"].append(
                    float(conf)
                )

                # --------------------------------------------------
                # Thermal evidence
                # --------------------------------------------------

                if thermal_used_this_frame:
                    record["thermal"] = True
                    record["source"] = "RGB+THERMAL"

                    score = thermal_box_score(
                        thermal_gray_current,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    record["thermal_score"] = max(
                        record["thermal_score"],
                        score
                    )

                # --------------------------------------------------
                # GPS
                # --------------------------------------------------

                telemetry_row = None

                if USE_TELEMETRY:
                    telemetry_row = (
                        get_telemetry_for_frame(
                            telemetry_df,
                            frame_idx,
                            fps
                        )
                    )

                    if telemetry_row is not None:
                        lat, lon = (
                            pixel_to_latlon(
                                cx,
                                cy,
                                telemetry_row,
                                width,
                                height,
                                HORIZONTAL_FOV_DEG
                            )
                        )

                        record["positions"].append(
                            (lat, lon)
                        )

                        # --------------------------------------------------
                        # Sonar
                        # --------------------------------------------------

                        if (
                            USE_SONAR
                            and sonar_df is not None
                        ):
                            (
                                matched,
                                sonar_conf,
                                sonar_type
                            ) = match_sonar(
                                lat,
                                lon,
                                telemetry_row[
                                    "timestamp"
                                ],
                                sonar_df
                            )

                            if matched:
                                record[
                                    "sonar_score"
                                ] = max(
                                    record[
                                        "sonar_score"
                                    ],
                                    sonar_conf
                                )

                                record[
                                    "sonar_target_type"
                                ] = sonar_type

                        # --------------------------------------------------
                        # UWB (close-range confirmation)
                        # --------------------------------------------------

                        if (
                            USE_UWB
                            and uwb_df is not None
                        ):
                            uwb_conf = match_uwb(
                                telemetry_row[
                                    "timestamp"
                                ],
                                uwb_df
                            )

                            record[
                                "uwb_score"
                            ] = max(
                                record[
                                    "uwb_score"
                                ],
                                uwb_conf
                            )

                # --------------------------------------------------
                # Best evidence
                # --------------------------------------------------

                if conf > record["best_conf"]:
                    record["best_conf"] = float(conf)

                    evidence_path = (
                        save_evidence_crop(
                            rgb_frame,
                            x1,
                            y1,
                            x2,
                            y2,
                            track_id,
                            frame_idx,
                            float(conf)
                        )
                    )

                    if evidence_path is not None:
                        record[
                            "best_frame"
                        ] = evidence_path

                # --------------------------------------------------
                # Confirmation
                # --------------------------------------------------

                record["confirmed"] = (
                    record["frames_seen"]
                    >= MIN_CONFIRMATION_FRAMES
                )

                # --------------------------------------------------
                # Priority
                # --------------------------------------------------

                record["priority_score"] = (
                    calculate_priority_score(
                        record["best_conf"],
                        record["frames_seen"],
                        record["positions"],
                        record["thermal_score"],
                        record["sonar_score"],
                        record["uwb_score"]
                    )
                )

                record["priority_level"] = (
                    priority_level(
                        record["priority_score"]
                    )
                )

                # --------------------------------------------------
                # Bounding box
                # --------------------------------------------------

                if record["confirmed"]:
                    status = "CONFIRMED SURVIVOR"
                    box_color = (
                        0,
                        255,
                        0
                    )
                else:
                    status = "POSSIBLE SURVIVOR"
                    box_color = (
                        0,
                        255,
                        255
                    )

                cv2.rectangle(
                    input_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                label = (
                    f"{status} "
                    f"ID:{track_id} "
                    f"{float(conf):.2f} "
                    f"P:{record['priority_score']:.0f}"
                )

                draw_label(
                    input_frame,
                    label,
                    x1,
                    y1,
                    box_color
                )

                info_text = (
                    f"{record['priority_level']} "
                    f"Frames:{record['frames_seen']} "
                    f"T:{record['thermal_score']:.2f} "
                    f"S:{record['sonar_score']:.2f} "
                    f"U:{record['uwb_score']:.2f}"
                )

                cv2.putText(
                    input_frame,
                    info_text,
                    (
                        x1,
                        min(
                            y2 + 24,
                            height - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2,
                    cv2.LINE_AA
                )

                if (
                    telemetry_row is not None
                    and record["positions"]
                ):
                    lat, lon = record[
                        "positions"
                    ][-1]

                    gps_text = (
                        f"GPS: "
                        f"{lat:.6f}, "
                        f"{lon:.6f}"
                    )

                    cv2.putText(
                        input_frame,
                        gps_text,
                        (
                            x1,
                            min(
                                y2 + 48,
                                height - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        box_color,
                        2,
                        cv2.LINE_AA
                    )

            # --------------------------------------------------
            # LIVE CSV UPDATE
            # --------------------------------------------------
            # The current frame has now been fully processed.
            # Save the latest survivor state immediately.
            update_live_survivor_csv(
                track_records
            )

            # --------------------------------------------------
            # Remove old tracks
            # --------------------------------------------------

            for tid in list(
                active_tracks.keys()
            ):
                if (
                    frame_idx
                    - active_tracks[tid]["frame"]
                    > MAX_TRACK_MISSING_FRAMES
                ):
                    del active_tracks[tid]

            # --------------------------------------------------
            # Dashboard counts
            # --------------------------------------------------

            confirmed_count = 0
            possible_count = 0
            critical_count = 0

            for record in track_records.values():
                if (
                    frame_idx
                    - record["last_seen_frame"]
                    > MAX_TRACK_MISSING_FRAMES
                ):
                    continue

                if record["confirmed"]:
                    confirmed_count += 1
                else:
                    possible_count += 1

                if (
                    record["priority_level"]
                    == "CRITICAL"
                ):
                    critical_count += 1

            total_unique = len(
                track_records
            )

            draw_dashboard(
                input_frame,
                confirmed_count,
                possible_count,
                total_unique,
                ENVIRONMENT,
                detection_mode,
                thermal_used_this_frame,
                USE_TELEMETRY,
                USE_SONAR and sonar_df is not None,
                critical_count,
                USE_UWB and uwb_df is not None
            )

            cv2.putText(
                input_frame,
                f"FRAME: {frame_idx}/{total_frames}",
                (
                    max(
                        20,
                        width - 310
                    ),
                    height - 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # --------------------------------------------------
            # Save 1080p output
            # --------------------------------------------------

            output_frame = fit_to_1080p(
                input_frame
            )

            output.write(
                output_frame
            )

            # --------------------------------------------------
            # Jupyter display
            #
            # NO cv2.imshow()
            # NO cv2.waitKey()
            # --------------------------------------------------

            if (
                frame_idx % DISPLAY_EVERY == 0
            ):
                display_frame_jupyter(
                    output_frame,
                    frame_idx,
                    total_frames
                )

            if frame_idx % 30 == 0:
                print(
                    f"Processed {frame_idx}/{total_frames} | "
                    f"Confirmed: {confirmed_count} | "
                    f"Possible: {possible_count} | "
                    f"Mode: {detection_mode}"
                )

            frame_idx += 1

    finally:
        rgb_cap.release()

        if thermal_cap is not None:
            thermal_cap.release()

        output.release()

        # IMPORTANT:
        # cv2.destroyAllWindows() is intentionally NOT used.
        # OpenCV was installed without HighGUI support in the
        # environment that produced the original error.

    return track_records


# ==============================================================
# 21. LOAD TELEMETRY
# ==============================================================

telemetry_df = None

if USE_TELEMETRY:
    print("\nLoading telemetry...")

    try:
        telemetry_df = load_telemetry(
            TELEMETRY_CSV
        )

        print(
            f"Telemetry rows: "
            f"{len(telemetry_df)}"
        )

    except Exception as error:
        print(
            f"WARNING: telemetry loading failed: {error}"
        )
        telemetry_df = None
        USE_TELEMETRY = False


# ==============================================================
# 22. LOAD SONAR
# ==============================================================

if USE_SONAR and USE_TELEMETRY:
    print("\nLoading sonar detections...")

    try:
        sonar_df = load_sonar(
            SONAR_CSV
        )

        print(
            f"Sonar rows: "
            f"{len(sonar_df)}"
        )

    except Exception as error:
        print(
            f"WARNING: sonar loading failed: {error}"
        )
        sonar_df = None


# ==============================================================
# 22B. LOAD UWB
# ==============================================================

if USE_UWB and USE_TELEMETRY:
    print("\nLoading UWB detections...")

    try:
        uwb_df = load_uwb(
            UWB_CSV
        )

        print(
            f"UWB rows: "
            f"{len(uwb_df)}"
        )

    except Exception as error:
        print(
            f"WARNING: UWB loading failed: {error}"
        )
        uwb_df = None


# ==============================================================
# 23. RUN
# ==============================================================

print("\nStarting advanced survivor detection...")
print(f"Environment: {ENVIRONMENT}")
print(f"Thermal: {USE_THERMAL}")
print(f"SAHI: {USE_SAHI and SAHI_AVAILABLE}")
print(f"P2: {P2_MODEL_AVAILABLE}")

track_records = (
    process_recorded_video()
)


# ==============================================================
# 24. GPS DEDUPLICATION
# ==============================================================

print(
    f"\nRaw tracks: "
    f"{len(track_records)}"
)

if USE_TELEMETRY:
    deduped_records = (
        dedup_by_distance(
            track_records
        )
    )

    print(
        "After GPS deduplication: "
        f"{len(deduped_records)}"
    )
else:
    deduped_records = dict(
        track_records
    )


# ==============================================================
# 25. GEOJSON + MAP
# ==============================================================

feature_collection = None

if USE_TELEMETRY:
    if geojson is not None:
        feature_collection = (
            build_triage_output(
                deduped_records
            )
        )

        save_geojson(
            feature_collection,
            OUTPUT_GEOJSON
        )

        if folium is not None:
            render_map(
                feature_collection,
                OUTPUT_MAP_HTML
            )
    else:
        print(
            "geojson is not installed. "
            "Run: %pip install geojson"
        )


# ==============================================================
# 26. FINAL RESULT
# ==============================================================

confirmed_survivors = 0
possible_survivors = 0
critical_survivors = 0

for record in deduped_records.values():
    if record["confirmed"]:
        confirmed_survivors += 1
    else:
        possible_survivors += 1

    if record["priority_level"] == "CRITICAL":
        critical_survivors += 1


print("\n")
print("==============================================")
print("       SURVIVOR DETECTION COMPLETE")
print("==============================================")
print(
    f"Confirmed survivors : "
    f"{confirmed_survivors}"
)
print(
    f"Possible survivors  : "
    f"{possible_survivors}"
)
print(
    f"Critical priority   : "
    f"{critical_survivors}"
)
print(
    f"Total unique tracks : "
    f"{len(deduped_records)}"
)
print(
    f"Output resolution   : "
    f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
)
print(
    f"Output video        : "
    f"{OUTPUT_VIDEO}"
)
print(
    f"Evidence folder     : "
    f"{EVIDENCE_DIR}"
)

if USE_TELEMETRY:
    print(
        f"GeoJSON             : "
        f"{OUTPUT_GEOJSON}"
    )
    print(
        f"Map                 : "
        f"{OUTPUT_MAP_HTML}"
    )

print("==============================================")
print("Processing finished successfully.")

# IMPORTANT:
# YOLO COCO class 0 detects "person".
# It does not medically or semantically prove that a person
# is a survivor. A disaster-specific trained model is required
# for true survivor classification.


# ==============================================================
# 27. SURVIVOR DETECTION RESULTS TABLE + INTERACTIVE MAP
# ==============================================================

# Build a clean table from the FINAL GPS-deduplicated survivor tracks.
survivor_rows = []

for tid, record in deduped_records.items():

    if not record.get("positions"):
        continue

    # Use the mean of the GPS positions for the final survivor position.
    latitude = float(np.mean([p[0] for p in record["positions"]]))
    longitude = float(np.mean([p[1] for p in record["positions"]]))

    survivor_rows.append({
        "ID": int(tid),
        "Confidence": round(float(record["best_conf"]), 2),
        "Latitude": round(latitude, 6),
        "Longitude": round(longitude, 6),
        "Priority": record.get("priority_level", "LOW"),
        "Priority Score": round(float(record.get("priority_score", 0.0)), 1),
        "Confirmed": bool(record.get("confirmed", False)),
        "Frames": int(record.get("frames_seen", 0))
    })

# Sort highest-confidence detections first and assign display IDs 1, 2, 3...
survivor_rows.sort(key=lambda x: x["Confidence"], reverse=True)

for display_id, row in enumerate(survivor_rows, start=1):
    row["ID"] = display_id

survivor_results_df = pd.DataFrame(
    survivor_rows,
    columns=[
        "ID",
        "Confidence",
        "Latitude",
        "Longitude",
        "Priority",
        "Priority Score",
        "Confirmed",
        "Frames"
    ]
)

print("\n")
print("==============================================================")
print("                 SURVIVOR DETECTION RESULTS")
print("==============================================================")

if survivor_results_df.empty:
    print("No survivor detections with valid GPS coordinates were found.")
else:
    # Display exactly the main fields requested.
    display(
        survivor_results_df[
            ["ID", "Confidence", "Latitude", "Longitude", "Priority"]
        ].style.format({
            "Confidence": "{:.2f}",
            "Latitude": "{:.6f}",
            "Longitude": "{:.6f}"
        })
    )

    # Also save the result table for reports/presentations.
    survivor_results_df.to_csv(
        LIVE_RESULTS_CSV,
        index=False
    )

    print("\nFull results saved to: survivor_detection_result.csv")


# ==============================================================
# 28. DISPLAY INTERACTIVE RESCUE MAP INSIDE JUPYTER
# ==============================================================

print("\n==============================================================")
print("                    INTERACTIVE RESCUE MAP")
print("==============================================================")

if USE_TELEMETRY and folium is not None and not survivor_results_df.empty:

    # Center map on all detected survivor coordinates.
    center_lat = survivor_results_df["Latitude"].mean()
    center_lon = survivor_results_df["Longitude"].mean()

    rescue_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        control_scale=True
    )

    # ----------------------------------------------------------
    # Add survivor markers
    # ----------------------------------------------------------
    for _, row in survivor_results_df.iterrows():

        priority = row["Priority"]

        if priority == "CRITICAL":
            marker_color = "red"
        elif priority == "HIGH":
            marker_color = "orange"
        elif priority == "MEDIUM":
            marker_color = "blue"
        else:
            marker_color = "green"

        popup_html = f"""
        <div style="font-family:Arial; width:230px;">
            <h4>🚨 SURVIVOR #{int(row['ID'])}</h4>
            <b>Confidence:</b> {row['Confidence']:.2f}<br>
            <b>Latitude:</b> {row['Latitude']:.6f}<br>
            <b>Longitude:</b> {row['Longitude']:.6f}<br>
            <b>Priority:</b> {priority}<br>
            <b>Priority Score:</b> {row['Priority Score']:.1f}/100<br>
            <b>Confirmed:</b> {row['Confirmed']}<br>
            <b>Frames:</b> {int(row['Frames'])}
        </div>
        """

        folium.Marker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            tooltip=(
                f"Survivor #{int(row['ID'])} | "
                f"{priority} | "
                f"{row['Confidence']:.2f}"
            ),
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            icon=folium.Icon(
                color=marker_color,
                icon="user",
                prefix="fa"
            )
        ).add_to(rescue_map)

    # ----------------------------------------------------------
    # Add drone telemetry flight path
    # ----------------------------------------------------------
    if telemetry_df is not None and not telemetry_df.empty:

        flight_path = [
            [float(row["lat"]), float(row["lon"])]
            for _, row in telemetry_df.iterrows()
        ]

        if len(flight_path) >= 2:
            folium.PolyLine(
                locations=flight_path,
                tooltip="Drone Flight Path",
                weight=3,
                opacity=0.7
            ).add_to(rescue_map)

        # Mark drone starting position.
        folium.Marker(
            location=flight_path[0],
            tooltip="Drone Start",
            icon=folium.Icon(
                color="black",
                icon="plane",
                prefix="fa"
            )
        ).add_to(rescue_map)

    # Fit map to all survivor points.
    bounds = [
        [float(v), float(w)]
        for v, w in zip(
            survivor_results_df["Latitude"],
            survivor_results_df["Longitude"]
        )
    ]

    if bounds:
        rescue_map.fit_bounds(bounds, padding=(30, 30))

    # Save HTML.
    rescue_map.save("interactive_rescue_map.html")

    # Display directly inside Jupyter Notebook.
    display(rescue_map)

    print("\nInteractive map saved to: interactive_rescue_map.html")

else:
    print(
        "Interactive map could not be displayed. "
        "Make sure telemetry.csv, folium, and valid GPS detections are available."
    )


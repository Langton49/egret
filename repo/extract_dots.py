"""
extract_dots.py

Extracts bird dot positions from TWI dotted screen captures and generates
YOLO-format label files paired with the corresponding high-res images.

Pipeline:
    1. Load screen capture
    2. Detect and crop out the count window UI panel
    3. Detect colored dot markers using HSV color thresholding
    4. Filter out boundary lines (elongated components) vs dots (compact)
    5. Scale dot pixel coordinates to high-res image space (25% zoom → 4x)
    6. Write YOLO label file: class x_center y_center width height (normalized)

Usage:
    # Dry run — show detection stats without writing labels
    python extract_dots.py --dry-run

    # Full extraction
    python extract_dots.py

    # Single image for debugging
    python extract_dots.py --debug path/to/screen_capture.JPG

Output:
    data/yolo/images/  <- symlinks or copies of high-res JPGs
    data/yolo/labels/  <- YOLO .txt label files
    data/yolo/extract_report.json  <- per-image stats and failures
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).parent
DOTTED_DIR = REPO_DIR / "data" / "dotted_images"
HIGHRES_DIR = REPO_DIR / "data" / "training_images"
LABELS_DIR = REPO_DIR / "data" / "yolo" / "labels"
IMAGES_DIR = REPO_DIR / "data" / "yolo" / "images"
REPORT_PATH = REPO_DIR / "data" / "yolo" / "extract_report.json"
AVIAN_CSV = REPO_DIR / "data" / "databases" / "processed" / "avianData20102021.csv.gz"

# ---------------------------------------------------------------------------
# YOLO class definitions
# All species collapsed to a single "bird" class for initial training.
# Set MULTI_CLASS = True to train species-level classes instead.
# ---------------------------------------------------------------------------
MULTI_CLASS = False

SPECIES_TO_CLASS = {
    # Brown Pelican
    "BRPE": 0,
    # Laughing Gull
    "LAGU": 1,
    # Great Egret
    "GREG": 2,
    # Tricolored Heron
    "TRHE": 3,
    # Roseate Spoonbill
    "ROSP": 4,
    # Royal Tern
    "ROYT": 5,
    # Sandwich Tern
    "SATE": 6,
    # Magnificent Frigatebird
    "MAFR": 7,
    # All others → generic waterbird
    "OTHER": 8,
}
SINGLE_CLASS = 0  # used when MULTI_CLASS = False

# ---------------------------------------------------------------------------
# HSV color ranges for each dot color category.
# Each entry: (lower_hsv, upper_hsv, species_hint)
# Hue is 0-179 in OpenCV, Saturation/Value 0-255.
# Red wraps around 0/180 so it has two ranges.
# ---------------------------------------------------------------------------
COLOR_RANGES = [
    # Cyan — MAFR, SATE
    {
        "name": "cyan",
        "ranges": [((85, 150, 150), (100, 255, 255))],
        "species_hint": ["MAFR", "SATE"],
    },
    # Red — BRPE Ad, LAGU Stand/Roost, TRHE (two HSV ranges for red)
    {
        "name": "red",
        "ranges": [
            ((0, 160, 120), (8, 255, 255)),
            ((168, 160, 120), (179, 255, 255)),
        ],
        "species_hint": ["BRPE", "LAGU", "TRHE"],
    },
    # Yellow — BRPE Imm
    {
        "name": "yellow",
        "ranges": [((22, 160, 160), (32, 255, 255))],
        "species_hint": ["BRPE"],
    },
    # Green — GREG, ROYT, SATE
    {
        "name": "green",
        "ranges": [((50, 130, 100), (80, 255, 220))],
        "species_hint": ["GREG", "ROYT", "SATE"],
    },
    # Magenta/Pink — LAGU, TRHE, ROSP
    {
        "name": "magenta",
        "ranges": [((143, 130, 130), (163, 255, 255))],
        "species_hint": ["LAGU", "ROSP", "TRHE"],
    },
    # Orange — some BRPE variants
    {
        "name": "orange",
        "ranges": [((10, 160, 160), (20, 255, 255))],
        "species_hint": ["BRPE"],
    },
    # White is removed — too many false positives on rock/sand backgrounds
]

# Dot detection parameters
MIN_DOT_AREA = 30         # pixels² — raised to reduce noise and text false positives
MAX_DOT_AREA = 1800       # pixels² — tightened; above this is a line segment or UI element
MAX_ASPECT_RATIO = 3.0    # width/height — tightened; dots are more compact than lines
MIN_CIRCULARITY = 0.15    # 0=line, 1=perfect circle — filters elongated line fragments
NMS_MIN_DIST = 12         # pixels — merge detections closer than this (duplicate suppression)
FIXED_BOX_RADIUS = 0.012  # YOLO normalized half-width/height for each label box


# ---------------------------------------------------------------------------
# Panel detection
# ---------------------------------------------------------------------------

def find_aerial_image_bounds(img):
    """
    Returns (x_end, y_start, y_end) defining the crop rectangle for the
    aerial image portion, excluding the count window UI panel.

    Handles two panel layouts:
    - 2010-2021: large fixed sidebar on the right (~25-35% of width)
    - 2023+: small floating window overlaid top-right

    Strategy: use the saturation channel — the aerial image has high saturation
    (vegetation, coloured markers) while the gray Windows UI panel has near-zero
    saturation. Find the rightmost column where mean saturation drops to near zero
    across a wide-enough strip to confirm it is the panel, not just a gap in
    the image content.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # Column-wise mean saturation
    col_sat = sat.mean(axis=0)

    PANEL_SAT_THRESH = 25    # panel columns have very low saturation
    PANEL_MIN_WIDTH = int(w * 0.12)  # panel must span at least 12% of width

    # Find the leftmost x where a contiguous low-saturation strip begins
    # and extends to the right edge
    best_x = w  # default: no panel found

    x = w - 1
    while x >= PANEL_MIN_WIDTH:
        if col_sat[x] < PANEL_SAT_THRESH:
            # Check if this low-saturation region extends at least PANEL_MIN_WIDTH
            x_start = x
            while x_start > 0 and col_sat[x_start - 1] < PANEL_SAT_THRESH:
                x_start -= 1
            strip_width = x - x_start + 1
            if strip_width >= PANEL_MIN_WIDTH:
                best_x = x_start
                break
        x -= 1

    return best_x, 0, h


def crop_to_aerial(img):
    x_end, y_start, y_end = find_aerial_image_bounds(img)
    return img[y_start:y_end, 0:x_end], x_end, y_start


# ---------------------------------------------------------------------------
# Dot detection
# ---------------------------------------------------------------------------

def _circularity(contour, area):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def _nms(detections, min_dist):
    """Remove duplicate detections closer than min_dist pixels, keeping one per cluster."""
    if not detections:
        return detections
    kept = []
    suppressed = set()
    for i, (x1, y1, c1) in enumerate(detections):
        if i in suppressed:
            continue
        kept.append((x1, y1, c1))
        for j, (x2, y2, _) in enumerate(detections):
            if j != i and j not in suppressed:
                if np.hypot(x2 - x1, y2 - y1) < min_dist:
                    suppressed.add(j)
    return kept


def detect_dots(img_crop):
    """
    Returns list of (x, y, color_name) in pixel coords of the cropped image.
    Filters out boundary lines and UI artefacts using area, aspect ratio,
    and circularity. Applies NMS to remove duplicate detections.
    """
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    detections = []

    for color_def in COLOR_RANGES:
        # Build combined mask for all ranges of this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in color_def["ranges"]:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask |= cv2.inRange(hsv, lo_arr, hi_arr)

        # Morphological cleanup — close small gaps, remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]

            if area < MIN_DOT_AREA or area > MAX_DOT_AREA:
                continue

            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > MAX_ASPECT_RATIO:
                continue  # elongated — likely a boundary line segment

            # Circularity check — isolate this component and find its contour
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            component_mask[labels == i] = 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                circ = _circularity(contours[0], area)
                if circ < MIN_CIRCULARITY:
                    continue  # too elongated — boundary line fragment

            cx, cy = centroids[i]
            detections.append((float(cx), float(cy), color_def["name"]))

    # Suppress duplicates from overlapping color ranges
    detections = _nms(detections, NMS_MIN_DIST)
    return detections


# ---------------------------------------------------------------------------
# Filename → high-res image path
# ---------------------------------------------------------------------------

def parse_source_filename(screen_capture_path):
    """
    Extract the source image filename from the screen capture filename or
    embedded title. Returns the base filename without extension.

    Screen capture filenames follow patterns like:
      7May2010DLJ1Area1.JPG          -> area capture, no direct photo link
      19May2018Area01_Cam1Card2-0042 -> Camera1 Card2 Photo 0042
      JuneArea20_24June2023_Cam1Card2_4833.tif

    Returns dict with keys: camera, card, photo (ints) when parseable, else None.
    """
    name = Path(screen_capture_path).stem

    # Pattern: Cam{N}Card{N}[-_]{photo}  or  Camera{N}-Card{N}-{photo}
    patterns = [
        r"[Cc]am(?:era)?(\d)[_\-]?[Cc]ard(\d+)[_\-](\d+)",
        r"Camera\s*(\d)[_\-]Card\s*(\d+)[_\-](\d+)",
        r"C(\d)C(\d+)[\-_](\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return {
                "camera": int(m.group(1)),
                "card": int(m.group(2)),
                "photo": int(m.group(3)),
            }
    return None


def find_highres_image(parsed, highres_root):
    """
    Given parsed camera/card/photo numbers, find the matching high-res JPG
    in the training_images directory tree.
    """
    if parsed is None:
        return None

    cam = parsed["camera"]
    card = parsed["card"]
    photo = parsed["photo"]

    # Search for files matching Camera{cam}*Card{card}*{photo}*
    # The exact directory structure varies by year/colony so we glob broadly
    patterns = [
        f"**/*Camera{cam}*Card{card}*{photo:04d}*.jpg",
        f"**/*Camera{cam}*Card{card}*{photo:04d}*.JPG",
        f"**/*Cam{cam}*Card{card}*{photo:04d}*.jpg",
        f"**/*Cam{cam}*Card{card}*{photo}*.jpg",
    ]
    for pat in patterns:
        matches = list(highres_root.glob(pat))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Label writing
# ---------------------------------------------------------------------------

def write_yolo_label(label_path, detections, img_w, img_h):
    """
    Write YOLO format label file.
    detections: list of (x_px, y_px, color_name) in cropped image coords
    """
    lines = []
    for (x, y, color) in detections:
        x_norm = x / img_w
        y_norm = y / img_h
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        cls = SINGLE_CLASS if not MULTI_CLASS else 0  # placeholder for multi-class
        lines.append(f"{cls} {x_norm:.6f} {y_norm:.6f} {FIXED_BOX_RADIUS} {FIXED_BOX_RADIUS}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def debug_visualise(screen_capture_path):
    """
    Show detection results overlaid on the screen capture for a single image.
    Saves annotated image to data/yolo/debug/
    """
    debug_dir = REPO_DIR / "data" / "yolo" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(screen_capture_path))
    if img is None:
        print(f"Could not load: {screen_capture_path}")
        return

    crop, x_end, y_start = crop_to_aerial(img)
    detections = detect_dots(crop)

    # Draw panel boundary
    vis = img.copy()
    cv2.line(vis, (x_end, 0), (x_end, img.shape[0]), (0, 255, 0), 2)

    # Draw detected dots on original image coordinates
    color_bgr = {
        "cyan": (255, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "magenta": (255, 0, 255),
        "orange": (0, 165, 255),
        "white": (200, 200, 200),
    }
    for (x, y, color) in detections:
        cx = int(x)
        cy = int(y + y_start)
        bgr = color_bgr.get(color, (128, 128, 128))
        cv2.circle(vis, (cx, cy), 8, bgr, 2)
        cv2.circle(vis, (cx, cy), 2, bgr, -1)

    out_path = debug_dir / (Path(screen_capture_path).stem + "_debug.jpg")
    cv2.imwrite(str(out_path), vis)
    print(f"Saved debug image: {out_path}")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  ({d[0]:.0f}, {d[1]:.0f}) color={d[2]}")


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(dry_run=False):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not DOTTED_DIR.exists():
        print(f"ERROR: dotted_images directory not found at {DOTTED_DIR}")
        sys.exit(1)

    # Collect all screen captures
    screen_captures = sorted(
        list(DOTTED_DIR.rglob("*.JPG")) + list(DOTTED_DIR.rglob("*.jpg"))
    )
    print(f"Screen captures found: {len(screen_captures)}")

    report = {
        "total": len(screen_captures),
        "processed": 0,
        "no_dots_detected": 0,
        "highres_not_found": 0,
        "errors": 0,
        "per_image": {},
    }

    for i, sc_path in enumerate(screen_captures):
        rel = sc_path.relative_to(DOTTED_DIR)
        label_stem = str(rel).replace(os.sep, "__").replace("/", "__")
        label_stem = Path(label_stem).stem

        try:
            img = cv2.imread(str(sc_path))
            if img is None:
                report["errors"] += 1
                continue

            crop, x_end, y_start = crop_to_aerial(img)
            crop_h, crop_w = crop.shape[:2]
            detections = detect_dots(crop)

            if len(detections) == 0:
                report["no_dots_detected"] += 1
                report["per_image"][label_stem] = {"dots": 0, "status": "no_dots"}
                continue

            # Find corresponding high-res image
            parsed = parse_source_filename(sc_path)
            highres_path = find_highres_image(parsed, HIGHRES_DIR) if parsed else None

            status = "ok"
            if highres_path is None:
                report["highres_not_found"] += 1
                status = "highres_not_found"
                # Still write label — will be matched manually or skipped later

            if not dry_run:
                label_path = LABELS_DIR / (label_stem + ".txt")
                write_yolo_label(label_path, detections, crop_w, crop_h)

                if highres_path is not None:
                    # Symlink high-res image into yolo/images/
                    img_link = IMAGES_DIR / (label_stem + highres_path.suffix)
                    if not img_link.exists():
                        try:
                            img_link.symlink_to(highres_path.resolve())
                        except Exception:
                            import shutil
                            shutil.copy2(highres_path, img_link)

            report["processed"] += 1
            report["per_image"][label_stem] = {
                "dots": len(detections),
                "highres": str(highres_path) if highres_path else None,
                "status": status,
                "panel_x": x_end,
            }

        except Exception as e:
            report["errors"] += 1
            report["per_image"][label_stem] = {"status": f"error: {e}"}

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(screen_captures)} processed...")

    # Summary
    print(f"\n--- Extraction complete ---")
    print(f"  Total screen captures : {report['total']}")
    print(f"  Processed             : {report['processed']}")
    print(f"  No dots detected      : {report['no_dots_detected']}")
    print(f"  High-res not found    : {report['highres_not_found']}")
    print(f"  Errors                : {report['errors']}")

    if not dry_run:
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dot annotations from TWI screen captures")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect dots and report stats without writing label files")
    parser.add_argument("--debug", metavar="PATH",
                        help="Run debug visualisation on a single screen capture")
    args = parser.parse_args()

    if args.debug:
        debug_visualise(args.debug)
    else:
        run_extraction(dry_run=args.dry_run)

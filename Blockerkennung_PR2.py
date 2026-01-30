import cv2
import numpy as np
import os
import json

SLIDER_STATE_FILE = "data/slider_settings.json"

image_path = "data/diff_color.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")
original = image.copy()

# Debug-Funktion zur Anzeige mehrerer Bildverarbeitungsschritte nebeneinander
def show_debug_pipeline(stages_dict, height=700, window_name="Debug Pipeline"):
    def to_bgr(img):
        return img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def resize(img, h):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))
    def label_image(img, label):
        labeled = img.copy()
        cv2.rectangle(labeled, (0, 0), (img.shape[1], 25), (0, 0, 0), -1)
        cv2.putText(labeled, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return labeled
    labeled_images = [label_image(resize(to_bgr(img), height), name) for name, img in stages_dict.items()]
    stacked = np.hstack(labeled_images)
    cv2.imshow(window_name, stacked)

# Hauptfarberkennung (dominante Farbe) in einem erkannten Block (Kontur)
def get_dominant_color_hsv(image, contour):
    def classify_by_hsv(h, s, v):
        if s < 50: return "gray"
        if h < 15 or h > 160: return "red"
        elif 15 <= h < 40: return "yellow"
        elif 90 <= h < 135: return "blue"
        else: return "other"

    # Maske erzeugen für die gegebene Kontur
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.mean(hsv, mask=mask)[:3]
    label = classify_by_hsv(h, s, v)

    # TODO: solve double block (edge-case) 
    # label == "other"
    
    return label

def save_slider_settings():
    sliders = {
        name: cv2.getTrackbarPos(name, "Tuning")
        for name in SLIDER_NAMES
    }
    with open(SLIDER_STATE_FILE, "w") as f:
        json.dump(sliders, f)

def load_slider_settings():
    if os.path.exists(SLIDER_STATE_FILE):
        with open(SLIDER_STATE_FILE, "r") as f:
            return json.load(f)
    return {}

SLIDER_NAMES = [
    "Canny low", "Canny high", "Poly epsilon %", "Blur Kernel",
    "Closing Kernel", "Min Area", "Max Area",
    "Min AR *10", "Max AR *10", "Pixel/cm"
]

slider_defaults = {
    "Canny low": 50,
    "Canny high": 150,
    "Poly epsilon %": 2,
    "Blur Kernel": 3,
    "Closing Kernel": 5,
    "Min Area": 400,
    "Max Area": 20000,
    "Min AR *10": 12,
    "Max AR *10": 60,
    "Pixel/cm": 10
}

# Fenster und Slider erstellen
cv2.namedWindow("Tuning")
loaded = load_slider_settings()

for name in SLIDER_NAMES:
    value = loaded.get(name, slider_defaults[name])
    maxval = 255 if "Canny" in name else 10000 if "Area" in name else 100
    cv2.createTrackbar(name, "Tuning", value, maxval if maxval > 0 else 1, lambda x: None)

# Bildverarbeitung bei jedem Slider-Update
def update(val):
    # Aktuelle Slider-Werte auslesen
    low = cv2.getTrackbarPos("Canny low", "Tuning")
    high = cv2.getTrackbarPos("Canny high", "Tuning")
    eps_percent = cv2.getTrackbarPos("Poly epsilon %", "Tuning") / 100.0
    blur_k = max(3, cv2.getTrackbarPos("Blur Kernel", "Tuning"))
    if blur_k % 2 == 0:
        blur_k += 1
    morph_k = max(1, cv2.getTrackbarPos("Closing Kernel", "Tuning"))
    min_area = cv2.getTrackbarPos("Min Area", "Tuning")
    max_area = cv2.getTrackbarPos("Max Area", "Tuning")
    min_ar = cv2.getTrackbarPos("Min AR *10", "Tuning") / 10
    max_ar = cv2.getTrackbarPos("Max AR *10", "Tuning") / 10
    px_per_cm = max(1, cv2.getTrackbarPos("Pixel/cm", "Tuning"))

    # Farbmasken generieren (HSV und LAB-basierte Schwellen)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    mask_red1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 100, 50), (180, 255, 255))
    mask_yellow = cv2.inRange(hsv, (15, 100, 100), (40, 255, 255))
    mask_blue = cv2.inRange(hsv, (90, 50, 30), (130, 255, 255))

    mask_hsv = cv2.bitwise_or(mask_red1, mask_red2)
    mask_hsv = cv2.bitwise_or(mask_hsv, mask_yellow)
    mask_hsv = cv2.bitwise_or(mask_hsv, mask_blue)

    mask_lab = cv2.inRange(L, 200, 255)
    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)

    # Maske anwenden, in Graustufen konvertieren und glätten
    masked_img = cv2.bitwise_and(original, original, mask=combined_mask)
    gray_base = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_base, (blur_k, blur_k), 0)
    edges = cv2.Canny(blur, low, high)

    # Morphologische Kantenschließung
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    display = original.copy()
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)

    all_blocks = []
    results = []

    new_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area * 1.5:
            mask_split = np.zeros_like(closed)
            cv2.drawContours(mask_split, [cnt], -1, 255, -1)
            subcontours, _ = cv2.findContours(mask_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_contours.extend(subcontours)
            continue

        if area < min_area or area > max_area:
            continue

        approx = cv2.approxPolyDP(cnt, eps_percent * cv2.arcLength(cnt, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w < h:
            angle += 90
            w, h = h, w

        if h == 0:
            continue

        ar = w / h
        if ar < min_ar or ar > max_ar:
            continue

        angle_to_y = angle % 180
        if angle_to_y > 90:
            angle_to_y = 180 - angle_to_y

        box = np.intp(cv2.boxPoints(((cx, cy), (w, h), angle)))
        cx, cy = int(cx), int(cy)
        radius_px = int(round(7 * px_per_cm))
        color_label = get_dominant_color_hsv(image, box)

        all_blocks.append({
            "box": box,
            "center": (cx, cy),
            "radius_px": radius_px,
            "angle": angle,
            "angle_to_y": angle_to_y,
            "aspect_ratio": ar,
            "color_label": color_label
        })

    contours.extend(new_contours)

    # Zeichne alle erkannten Blöcke ein und prüfe auf Überlappung (Kollision)
    for i, blk in enumerate(all_blocks):
        cx, cy = blk["center"]
        radius = blk["radius_px"]
        box = blk["box"]
        angle = blk["angle"]
        angle_to_y = blk["angle_to_y"]
        color_label = blk["color_label"]

        collision = any(
            i != j and cv2.pointPolygonTest(other["box"], (cx, cy), True) >= -radius
            for j, other in enumerate(all_blocks)
        )

        # Kollision Styling
        outline_color = (0, 0, 255) if collision else (0, 255, 0)
        circle_color = (0, 255, 255) if collision else (255, 0, 0)
        cv2.drawContours(display, [box], -1, outline_color, 2)
        cv2.circle(display, (cx, cy), radius, circle_color, 2, lineType=cv2.LINE_AA)

        # Achsenrichtung und Beschriftung
        angle_rad = np.deg2rad(angle)
        x_axis = (int(cx + 40 * np.cos(angle_rad)), int(cy + 40 * np.sin(angle_rad)))
        y_axis = (int(cx - 40 * np.sin(angle_rad)), int(cy + 40 * np.cos(angle_rad)))

        cv2.line(display, (cx, cy), x_axis, (0, 0, 255), 2)
        cv2.line(display, (cx, cy), y_axis, (0, 255, 0), 2)
        cv2.putText(display, f"{angle_to_y:.1f}°", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display, color_label, (cx + 10, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"r={radius}px", (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        # Sammeln der Ergebnisse
        results.append({
            "center_x": cx,
            "center_y": cy,
            "rotation_deg_y": angle_to_y,
            "aspect_ratio": blk["aspect_ratio"],
            "color": color_label,
            "collision": collision
        })

    # Zeige Verarbeitungsschritte & Ergebnisbild
    show_debug_pipeline({
        "Original": original,
        "Masked": masked_img,
        "Blur": blur,
        "Canny": edges,
        "Closed": closed
    })
    cv2.imshow("Blockerkennung", display)

    # Ergebnisse als CSV-Datei exportieren
    with open("data/live_blocks_output.csv", "w") as f:
        f.write("center_x,center_y,rotation_deg_y,aspect_ratio,color,collision\n")
        for r in results:
            f.write(f"{r['center_x']},{r['center_y']},{r['rotation_deg_y']:.1f},{r['aspect_ratio']:.2f},{r['color']},{int(r['collision'])}\n")
# Erstes Update beim Start auslösen
update(0)
print("Drücke ESC um das Tuning der Erkennung zu verlassen..")
last_values = {}

# Änderungen an Slidern überwachen
while True:
    changed = False
    current_values = {}

    for name in SLIDER_NAMES:
        val = cv2.getTrackbarPos(name, "Tuning")
        current_values[name] = val
        if last_values.get(name) != val:
            changed = True

    if changed:
        update(0)
        last_values = current_values.copy()

    key = cv2.waitKey(100) & 0xFF
    if key == 27:   # ESC zum Beenden
        save_slider_settings()
        print("live_blocks_output.csv wurde gespeichert.")
        break

cv2.destroyAllWindows()
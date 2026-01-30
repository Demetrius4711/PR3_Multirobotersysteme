import cv2
import os
import numpy as np
# Pfad zur Datei
filename = r'C:\Users\flona\Desktop\PR3_test\Code_Prep\color_20260124_151350_812936.png'

# 1. Prüfe, ob die Datei existiert
if not os.path.exists(filename):
    print(f"❌ Fehler: Datei nicht gefunden: {filename}")
    exit()

# 2. Prüfe die Dateigröße
size = os.path.getsize(filename)
if size == 0:
    print(f"❌ Fehler: Datei ist leer (0 Byte): {filename}")
    exit()

# 3. Lade das Bild
image = cv2.imread(filename)

if image is None:
    print(f"❌ Fehler: Bild konnte nicht geladen werden. Prüfe Datei: {filename}")
    print("Mögliche Ursachen:")
    print("  - Falscher Pfad")
    print("  - Beschädigte Datei")
    print("  - Falsche Dateiendung (z. B. .png statt .PNG)")
    exit()

# 4. Jetzt sicher: image ist gültig
print(f"✅ Bild erfolgreich geladen: {image.shape}")

# 5. Kanten erkennen
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# 6. Hough-Transformation
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 7. Zeichne Linien und berechne Länge
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        print(f"Linienlänge: {length:.2f} Pixel")

        # Zeichne Linie
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 8. Zeige das Ergebnis
cv2.imshow('Kanten mit Länge', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
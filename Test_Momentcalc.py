import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Bild laden und vorverarbeiten
def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler: Bild {image_path} konnte nicht geladen werden!")
        return None
    
    # Größe anpassen für bessere Performance
    image = cv2.resize(image, (800, 600))
    
    # Rauschreduktion
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # In HSV-Farbraum konvertieren
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return image, hsv

# 2. Rote Farbe segmentieren
def detect_red_mask(hsv):
    # Untere und obere Grenzen für Rot im HSV-Raum
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Masken erstellen und kombinieren
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphologische Operationen zum Glätten
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return cleaned_mask

# 3. Konturen finden und Mittelpunkt berechnen
def find_center(image, mask):
    # Konturen finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Keine Konturen gefunden!")
        return image, None
    
    # Größte Kontur auswählen
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Kontur zeichnen
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    
    # Mittelpunkt mit Momenten berechnen
    M = cv2.moments(largest_contour)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)
        
        # Mittelpunkt markieren
        cv2.circle(image, center, 8, (0, 0, 255), -1)
        cv2.putText(image, f"Center: ({cx}, {cy})", (cx - 70, cy - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return image, center
    else:
        print("M00 ist Null - kann Mittelpunkt nicht berechnen")
        return image, None

# 4. Hauptfunktion
def main():
    # Bildpfad - HIER DEINEN EIGENEN PFAD EINFÜGEN!
    image_path = "color_20260124_151350_812936.png"
    
    # Bild laden und vorverarbeiten
    image, hsv = load_and_preprocess(image_path)
    if image is None:
        return
    
    # Rote Maske erstellen
    red_mask = detect_red_mask(hsv)
    
    # Mittelpunkt finden und anzeigen
    result_image, center = find_center(image.copy(), red_mask)
    
    if center is not None:
        print(f"2D-Mittelpunkt gefunden bei: {center}")
    
    # Ergebnisse anzeigen
    plt.figure(figsize=(15, 10))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Originalbild")
    
    plt.subplot(132)
    plt.imshow(red_mask, cmap='gray')
    plt.title("Rote Maske")
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Ergebnis mit Mittelpunkt")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
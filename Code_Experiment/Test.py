import pyrealsense2 as rs
import cv2
import numpy as np

# Pipeline initialisieren
pipeline = rs.pipeline()
config = rs.config()

# Streams konfigurieren (Farb- und Tiefenbild)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Pipeline starten
pipeline.start(config)

try:
    while True:
        # Frames abwarten
        frames = pipeline.wait_for_frames()
        
        # Farb- und Tiefen-Frames extrahieren
        color_frame = frames.get_color_frame()
    
        
        if not color_frame:
            continue
        
        # Konvertiere zu OpenCV-Format
        color_image = np.asanyarray(color_frame.get_data())
        
        
        
        
        # Bilder anzeigen
        cv2.imshow('Color', color_image)
        
        # Beenden mit 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Pipeline stoppen
    pipeline.stop()
    cv2.destroyAllWindows()

"""
Idee/ Pipeline: Zwei Funktionen die seperat aufgerufen werden können 

Übergeben wird dann das Video/ Bild

    Bildprep:
        -Genauer Aufbau (Funktionen):
            - Init und config Realsense
            - Erstellen eines Alignobjektes
            - Pipeline starten
            - Input + align
            - Convert Input Frames (depth/ color) to Numpy Array
            - Anzeigen der Bilder
            - Exit Funktion
            - Pipeline Stoppen

        - ohne cam:
            - Für die Testdaten müssen alignte Bilder erstellt werden 
                - TODO: Hackathon2: Testdaten aligend aufnehmen 

    Kantendedektion:
        -Ideen/ Grobe Pipeline:    
            - BGR 2 HSV
            - Farbsegmentierung jeweils nach der Farbe der Wahl 1.BSP wird Rot sein (0-10//170-180 im HSV Raum)
            - Morphologische Transformation  (Rauschen/ Löcher Schließen)
            - Konturensuche 
            - Kanten des Blocks anzeigen

        
    Berechnung des Mittelpunktes:
    (- Mit hilfe von 2D und 3D informationen (Verweundung des Bildmomentes))
        - Geg: 
            - Konturen der Blöcke 
        - Berechnung des Bildmomentes
        - Daraus den Schwerpunkt


"""

import cv2
import matplotlib  
import numpy as np
import os
import sys 
import pyrealsense2 as rs 

"""
--------------------------------------------
Sammlung von Funktionen
--------------------------------------------
"""


"""------- FKTs: Bildprep --------------"""
# Init und config Realsense
def Cam_Config():
    pipeline = rs.pipeline()
    config = rs.config()
    print("Init done")
    # Streams konfigurieren
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    print("config done")
    return pipeline, config


# erstellen eines Alignobjektes
def create_align_objekt():
    
    aligne_to = rs.stream.color
    print("align done")
    return rs.align(aligne_to)

# Pipeline starten
def start_pipeline(pipeline, config):
    profile = pipeline.start(config)
    print("RealSense funktioniert")
    #ToDo: abbruchfunktion und Fehlermeldung
    return profile 

# Input + align
def Input_Align(align_objekt, pipeline):
    frames = pipeline.wait_for_frames()
    aligned_frames = align_objekt.process(frames)
 

    depth_fr = aligned_frames.get_depth_frame()
    color_fr = aligned_frames.get_color_frame()
    return depth_fr, color_fr

#Convert Input Frames (depth/ color) to Numpy Array
def Convert_Inp_To_NP(depth_fr, color_fr):
    #abbruchfunktion wenn kein Frame
    if not depth_fr or not color_fr: 
        return None, None
    
    #normale umwandlung
    depth_im = np.asanyarray(depth_fr.get_data())
    color_im = np.asanyarray(color_fr.get_data())
    print("Convert to np done")
    return depth_im, color_im

# Depth filter zum Ausschluss bestimmter tiefen 
# Filter Pick up: 0-300
# Filter Pick up: 
def Depth_filter(min_depth, max_depth, depth_im):
    depth_mm = depth_im.astype(np.float32)
    
    valid_mask = depth_im > 0
    range_mask = (depth_mm >= min_depth) & (depth_mm <= max_depth)
    final_mask = valid_mask & range_mask
    return final_mask  

"""
#Anzeigen der Bilder
def show_img(masked_depth_im, color_im):
    #Anzeigen der Bilder
    if color_im is not None and masked_depth_im is not None:
        #Depth Image farbig darstellen
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(masked_depth_im, alpha=0.03), 
            cv2.COLORMAP_JET

        )
        #Bilder kombinieren 
        combined_image = np.hstack((color_im, depth_colormap))
        cv2.imshow("RealSense ColorImage with Depthcolormap", combined_image)

        return True
    return False 
"""
# Anzeigen der Bilder
def show_img(masked_depth_im, color_im, depth_im):
    if color_im is not None and masked_depth_im is not None:
        # 1. Farbbild nur anzeigen, wo die Maske True ist
        filtered_color = np.where(masked_depth_im[..., np.newaxis], color_im, 0)

        # 2. Tiefenbild für Visualisierung: Nur gültige Werte im Bereich
        depth_mm = depth_im.astype(np.float32)
        # Erstelle ein Tiefenbild nur für den Bereich (nicht die Maske!)
        depth_for_display = np.where(masked_depth_im, depth_mm, 0)  # Nur im Bereich, sonst 0

        # 3. Farbvisualisierung (nur für Anzeige!)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_for_display, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 4. Bilder nebeneinander zeigen
        combined_image = np.hstack((filtered_color, depth_colormap))
        cv2.imshow("RealSense ColorImage with Depthcolormap", combined_image)
        
        return True, combined_image
    return False


# Exit Funktion 
def Exit_Fkt():
    return cv2.waitKey(1) & 0xFF == ord('q')

# Pipeline Stoppen
def stop_pipeline(pipeline):
    pipeline.stop()
    cv2.destroyAllWindows()
 


"""------- FKTs: Kantendetektion --------------"""

# Bilateral Blur (Noice Reduction without detaillost) 
def Blur_Fkt(combined_img, Advanced):
    if Advanced == 0:
        Blur_img = cv2.bilateralFilter(combined_img, (15,15))
    #Blur_Image = cv2.cvtColor(Blur_filter, cv2.COLOR_BGR2RGB)
    else: 
        blurred = cv2.bilateralFilter(combined_img, (15,15))
        Blur_img = cv2.equalizeHist(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

    return Blur_img

# Kantenerkennung: 
def detect_edges(Blur_img, min, max):
    if len(Blur_img.shape) == 3:
        gray = cv2.cvtColor(Blur_img, cv2.COLOR_BGR2GRAY)
    else: 
        gray = Blur_img
    
    Edges = cv2.Canny(gray, min,max, apertureSize=3)

    return Edges

# Konturen Erkennen:
def find_contours(edges):
    contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contour
    

# Konturen Analysieren 
def analysze_contour(contours, min_area = 1000):
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            valid_contours.append(contour)
        
    return valid_contours





"""
--------------------------------------------
Hauptfunktion
--------------------------------------------
"""
def CamPipeline():

    # Init und config Realsense
    pipeline, config = Cam_Config()
    align = create_align_objekt()

    try: 
        profile = start_pipeline(pipeline, config)

        
        while True: 
            depth_frame, color_frame = Input_Align(align, pipeline)

            #depth_mask = Depth_filter(0, 100, depth_frame)

            depth_image, color_image = Convert_Inp_To_NP(depth_frame, color_frame)

            depth_mask = Depth_filter(0, 300, depth_image)
            
            if not show_img(depth_mask , color_image, depth_image):
                break
            else:
                show_img(depth_mask , color_image, depth_image)

            if Exit_Fkt():
                break 
    
    finally:     
        stop_pipeline(pipeline)




"""
--------------------------------------------
Hauptprogramm
--------------------------------------------
"""

CamPipeline()
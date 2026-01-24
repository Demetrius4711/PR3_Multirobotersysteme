#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import numpy as np
import cv2
import statistics

Bilder_anzeigen = False
Tiefenbild_anzeigen = False

fig_konturen_bild,axs_konturen_bild = plt.subplots()
axs_konturen_bild.set_title('HSV Bild')


# Schwellwerte für Farbfilter definieren (HSV)
Farb_Grenzen = {
    "Blau": {
        "U_Grenze": np.array([75, 110, 20]),
        "O_Grenze": np.array([105, 270, 70]),
    },
    "Gelb": {
        "U_Grenze": np.array([25, 170, 90]),
        "O_Grenze": np.array([35, 255, 180]),
    },
    "Rot": {
        "U_Grenze": np.array([0,80,70]),
        "O_Grenze": np.array([20,265,150]),
    },

}

# Funktion: Konturen finden und Abstand bestimmen (Berechnung zum Teil noch fehlerhaft)
def abstandMessen():
     # Variablen
    center = (0, 0)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    print(f'Dimension depth array:{depth_image.shape}')

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Bild einlesen
    BildOrgBGR = color_image

    # HSV Bild erstellen
    BildHSV = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # HSV Bild anzeigen
    if Bilder_anzeigen:
        fig = plt.figure()
        fig.canvas.manager.set_window_title('HSV Bild')
        plt.imshow(BildHSV)
        plt.show()

    AlleSegmentierungen = []

    # Segmentierung alle Farben
    for farbe, hsv_range in Farb_Grenzen.items():
        Teilsegmentierung = cv2.inRange(
            BildHSV, hsv_range["U_Grenze"], hsv_range["O_Grenze"])
        AlleSegmentierungen.append(Teilsegmentierung) 
        
        # Seg Bild anzeigen
        if Bilder_anzeigen:
            fig = plt.figure()
            fig.canvas.manager.set_window_title('Teil Seg Bild')
            plt.imshow(Teilsegmentierung)
            plt.show()

    # Alle Segmentierungen vereinen
    BildSeg = AlleSegmentierungen[0]
    for i in range(1, len(AlleSegmentierungen)):
        BildSeg = cv2.bitwise_or(BildSeg, AlleSegmentierungen[i])

    #print(f'Dimension seg array:{BildSeg.shape}')
    #print(f'BildSeg Array:{BildSeg}')
    
    # Seg Bild anzeigen
    if True:
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Seg Bild')
        plt.imshow(BildSeg)
        plt.show()

    # Finden von Konturen
    Konturen = cv2.findContours(
        BildSeg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Konturen = Konturen[0]

    # Konturen einzeichnen
    BildBGRKont = BildOrgBGR.copy()
    cv2.drawContours(BildBGRKont, Konturen, -1, (255, 0, 0), 3)

    #print('Anzahl der Konturen:', len(Konturen))

    abstand = 0

    # Prüfen, ob Konturen gefunden wurden
    if len(Konturen) != 0:

        max_Flaeche_Kontur = 0

        # Größte Fläche herausfinden
        for contour in Konturen:
            # Aktuelle Fläche berechnen
            current_Flaeche = cv2.contourArea(contour)

            # Aktualiseren, falls aktuelle Fläche größer als gespeicherte Fläche
            if current_Flaeche > max_Flaeche_Kontur:
                max_Flaeche_Kontur = current_Flaeche

        #print('Größte Fläche:', max_Flaeche_Kontur)
        
        # Fläche berechnen und Mittelpunkt einzeichnen
        for Kontur in Konturen:
            # Fläche der Kontur berechnen
            Flaeche_Kontur = cv2.contourArea(Kontur)

            # Mittelpunkt einzeichnen
            if  Flaeche_Kontur  == max_Flaeche_Kontur :

                # Tiefenbild anzeigen
                if Tiefenbild_anzeigen:
                    # Abstand zum Tisch festlegen (Variable wird nur zum Einfärben des Tiefenbilds, nicht aber für die Abstandberechnung verwendet)
                    Abstand_zum_Tisch = 300
                    print('Abstand zum Tisch:' + str(Abstand_zum_Tisch))

                    #
                    height, width = depth_image.shape
                    colored_depth = np.zeros((height, width, 3), dtype= np.uint8)

                    # Farben für Abstandsbereiche definieren
                    depth_colors =[
                        (Abstand_zum_Tisch - 30, Abstand_zum_Tisch, (255, 0, 0)),
                        (Abstand_zum_Tisch - 60, Abstand_zum_Tisch - 30, (255, 255, 0)),
                        (Abstand_zum_Tisch - 90, Abstand_zum_Tisch - 60, (0, 0, 255)),
                        (Abstand_zum_Tisch - 120, Abstand_zum_Tisch - 90, (255, 255, 0)),
                    ] 

                    # Farben zuordnen
                    for min_d, max_d, color in depth_colors:
                        mask = (depth_image >= min_d) & (depth_image <= max_d)
                        colored_depth[mask] = color

                    # Bild anzeigen
                    fig = plt.figure()
                    fig.canvas.manager.set_window_title('Tiefenbild')
                    plt.imshow(colored_depth)
                    plt.show()

                
                abstand_tisch = 0
                abstand_objekt = 0

                # Abstand zu Tisch bestimmen bestimmen
                speicher =([])
                x, y = depth_image.shape
                for i in range(x):
                    for j in range(y):
                        if BildSeg[i][j] == 0:
                            speicher.append(depth_image[i][j])

                speicher_sortiert = sorted(speicher)
                speicher_sortiert =[x for x in speicher_sortiert if x!= 0 and x<1000]
                #print(f'Speicher sortiert ohne null:{speicher_sortiert} ')

                if speicher_sortiert !=[]:  
                    abstand_tisch = statistics.mode(speicher_sortiert)


                # Abstand zu Objekt bestimmen bestimmen
                speicher =([])
                x, y = depth_image.shape
                for i in range(x):
                    for j in range(y):
                        if BildSeg[i][j] == 255:
                            speicher.append(depth_image[i][j])

                speicher_sortiert = sorted(speicher)
                speicher_sortiert =[x for x in speicher_sortiert if x!= 0 and x<1000]
                #print(f'Speicher sortiert ohne null:{speicher_sortiert} ')

                if speicher_sortiert !=[]:  
                    abstand_objekt = statistics.mode(speicher_sortiert)    

                    

    return(abstand_tisch, abstand_objekt)

# Funktion: Konturen finden und Mittelpunkt einzeichnen
def konturenFinden():
    # Variablen
    center = (0, 0)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Bild einlesen
    BildOrgBGR = color_image

    # HSV Bild erstellen
    BildHSV = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # HSV Bild anzeigen
    if Bilder_anzeigen:
        fig = plt.figure()
        fig.canvas.manager.set_window_title('HSV Bild')
        plt.imshow(BildHSV)
        plt.show()

    AlleSegmentierungen = []

    # Segmentierung alle Farben
    for farbe, hsv_range in Farb_Grenzen.items():
        Teilsegmentierung = cv2.inRange(
            BildHSV, hsv_range["U_Grenze"], hsv_range["O_Grenze"])
        AlleSegmentierungen.append(Teilsegmentierung) 
        
        # Seg Bild anzeigen
        if Bilder_anzeigen:
            fig = plt.figure()
            fig.canvas.manager.set_window_title('Teil Seg Bild')
            plt.imshow(Teilsegmentierung)
            plt.show()

    # Alle Segmentierungen vereinen
    BildSeg = AlleSegmentierungen[0]
    for i in range(1, len(AlleSegmentierungen)):
        BildSeg = cv2.bitwise_or(BildSeg, AlleSegmentierungen[i])

    # Finden von Konturen
    Konturen = cv2.findContours(
        BildSeg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Konturen = Konturen[0]

    # Konturen einzeichnen
    BildBGRKont = BildOrgBGR.copy()
    cv2.drawContours(BildBGRKont, Konturen, -1, (255, 0, 0), 3)

    #print('Anzahl der Konturen:', len(Konturen))

    # Prüfen, ob Konturen gefunden wurden
    if len(Konturen) != 0:

        max_Flaeche_Kontur = 0

        # Größte Fläche herausfinden
        for contour in Konturen:
            # Aktuelle Fläche berechnen
            current_Flaeche = cv2.contourArea(contour)

            # Aktualiseren, falls aktuelle Fläche größer als gespeicherte Fläche
            if current_Flaeche > max_Flaeche_Kontur:
                max_Flaeche_Kontur = current_Flaeche

        Mittelpunkte = []
        Drehwinkel =[] 

        # Fläche berechnen und Mittelpunkt einzeichnen
        for Kontur in Konturen:
            # Fläche der Kontur berechnen
            Flaeche_Kontur = cv2.contourArea(Kontur)

            # Drehwinkel der Kontur berechnen
            Rect = cv2.minAreaRect(Kontur)
            Winkel_Kontur = Rect[2]
            
            # Winkel als Wert zwischen -180 und 180 Grad ausgeben
            if Rect[1][0] < Rect [1][1]:
                Winkel_Kontur = Winkel_Kontur + 90
            if Winkel_Kontur > 180:
                Winkel_Kontur -= 360
            elif Winkel_Kontur < -180:
                Winkel_Kontur += 360
               
            # Mittelpunkt einzeichnen
            if  Flaeche_Kontur  == max_Flaeche_Kontur :

                # Schwerpunkt der Kontur bestimmen
                M = cv2.moments(Kontur)

                # Mittelpunkt einzeichnen
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    center = (center_x, center_y)

                    # Mittelpunkt und Winkel zu Liste hinzufügen
                    Mittelpunkte.append(center)
                    Drehwinkel.append(Winkel_Kontur)

                    # Größe des Kreuzes
                    cross_size = 80

                    # Horizontale Linie einzeichnen
                    cv2.line(BildBGRKont, (center_x - cross_size, center_y),
                             (center_x + cross_size, center_y), (0, 0, 0), 2)

                    # Vertikale Linie einzeichnen
                    cv2.line(BildBGRKont, (center_x, center_y - cross_size),
                             (center_x, center_y + cross_size), (0, 0, 0), 2)

        # Konturenbild anzeigen
        axs_konturen_bild.imshow(BildBGRKont)
        plt.show(block=False)
        plt.pause(1E-3)

    

    #print(f'Koordinaten der Mittelpunkte: {Mittelpunkte}')
    return(Mittelpunkte[0], Drehwinkel[0])


# Cobot einrichten
current_file = Path(__file__).resolve()

parent_dir = current_file.parents[0]

sys.path.append(str(parent_dir))

sys.path.append(str('../Schreibtisch/Projekt/xArm_Python_SDK/'))

from xarm.wrapper import XArmAPI # type: ignore

# ip Adresse
ip = '10.0.0.227'

#define arm
arm = XArmAPI(ip)

#connect and enable
arm.connect()
arm.motion_enable(enable=True)

#clear any previous warn codes
arm.clean_warn()
arm.clean_error()

#set mode and state
arm.set_mode(0)
arm.set_state(0)

# Greifer initialiseren
code = arm.set_gripper_mode(0)
code = arm.set_gripper_enable(True)

# Hauptprogramm
Gefunden = False
while not Gefunden:
    
    # Koordinaten der Bildmitte
    Mitte_X = 320
    Mitte_Y = 240

    # Initiale Position, Greifer öffnen
    code = arm._arm.set_servo_angle(angle=[0.5, -48.3, -53.8, 1.3, 102.1, 0.7], wait=False, radius=0.0)
    code = arm._arm.set_gripper_position(850, wait=True, speed=5000, auto_enable=True)

    # Variablen übergeben
    Mittelpunkt, Drehwinkel = konturenFinden()
    x_mitte, y_mitte = Mittelpunkt
    
    print(f'Mittelpunkt:{Mittelpunkt} und Drehwinkel: {Drehwinkel} ')

    # Wenn Mittelpunkt nicht gefunden -> Schleife erneut starten
    if Mittelpunkt == (0,0):
        continue

    # Schrittweise Ausrichtung des Cobots in x un y Richtung
    for i in range(3):
        # in X Richtung verfahren, große Schritte
        if x_mitte > Mitte_X+50:
            while(int(x_mitte) > Mitte_X+50):
                code = arm._arm.set_position(y=-30, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        elif x_mitte < Mitte_X-50:
            while(x_mitte < Mitte_X-50):
                code = arm._arm.set_position(y=30, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        # in X Richtung verfahren, kleine Schritte
        elif x_mitte > Mitte_X+10:
            while(x_mitte > Mitte_X+10):
                code = arm._arm.set_position(y=-5, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        elif x_mitte < Mitte_X-10:
            while(x_mitte < Mitte_X-10):
                code = arm._arm.set_position(y=5, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        # in Y Richtung verfahren, große Schritte
        if y_mitte > Mitte_Y+50:
            while(y_mitte > Mitte_Y+50):
                code = arm._arm.set_position(x=-30, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        elif y_mitte < Mitte_Y-50:
            while(y_mitte < Mitte_Y-50):
                code = arm._arm.set_position(x=30, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        #  in Y Richtung verfahren, kleine Schritte
        elif y_mitte > Mitte_Y+10:
            while(y_mitte > Mitte_Y+10):
                code = arm._arm.set_position(x=-5, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

        elif y_mitte < Mitte_Y-10:
            while(y_mitte < Mitte_Y-10):
                code = arm._arm.set_position(x=5, radius=-1, relative=True, wait=True)
                Mittelpunkt, Drehwinkel = konturenFinden()
                x_mitte, y_mitte = Mittelpunkt

    Gefunden = True

    # Höhe ausgeben (Berechnung zum Teil noch fehlerhaft)
    abstand_tisch , abstand_objekt = abstandMessen()
    print(f'Abstand zum Objekt:{abstand_objekt} ,Abstand zum Tisch:{abstand_tisch}')

    # Greifer mittig ausrichten
    code = arm._arm.set_position(x=78, radius=-1, relative=True, wait=True)
    code = arm._arm.set_position(y=37, radius=-1, relative=True, wait=True)
    position = arm.get_position()

    # Greifer auf richtige Positon drehen
    winkel = arm._arm.get_servo_angle()
    print(f'Servo Winkel: {winkel} ')
    code = arm._arm.set_servo_angle(angle=[winkel[1][0], winkel[1][1],winkel[1][2],winkel[1][3],winkel[1][4], winkel[1][5]+Drehwinkel-90], wait=False, radius=0.0)

    # Objekt greifen und Greifer neu ausrichten
    arm.set_position(z=10)
    code = arm._arm.set_gripper_position(500, wait=True, speed=5000, auto_enable=True)
    arm.set_position(z=50, relativ=True)
    arm.set_position(yaw=0)

    # Objekt zu Ablageort bringen und ablegen
    arm.set_position(x=0 , y=460 ,z=10)
    code = arm._arm.set_gripper_position(800, wait=True, speed=5000, auto_enable=True)
    arm.set_position(z=50, relativ=True)
    code = arm._arm.set_servo_angle(angle=[0.5, -48.3, -53.8, 1.3, 102.1, 0.7], wait=False, radius=0.0)





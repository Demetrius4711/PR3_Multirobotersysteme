
"""
Idee/ Pipeline: Zwei Funktionen die seperat aufgerufen werden können 

Übergeben wird dann das Video/ Bild

    Kantendedektion:    
        - BGR 2 HSV
        - Farbsegmentierung jeweils nach der Farbe der Wahl 1.BSP wird Rot sein (0-10//170-180 im HSV Raum)
        - Morphologische Transformation  (Rauschen/ Löcher Schließen)
        - Konturensuche 
        - Kanten des Blocks anzeigen
        








"""

import cv2 as cv
import matplotlib  
import numpy
import os
import sys 

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

image_hsv = None
image_src = None
pixel = (0,0,0) #RANDOM DEFAULT VALUE

ftypes = [
    ("JPG", "*.jpg;*.JPG;*.JPEG"), 
    ("PNG", "*.png;*.PNG"),
    ("GIF", "*.gif;*.GIF"),
    ("All files", "*.*")
]

def check_boundaries(value, tolerance, ranges, upper_or_lower):
    if ranges == 0:
        # set the boundary for hue
        boundary = 180
    elif ranges == 1:
        # set the boundary for saturation and value
        boundary = 255

    if(value + tolerance > boundary):
        value = boundary
    elif (value - tolerance < 0):
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + tolerance
        else:
            value = value - tolerance
    return value


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        # Set range = 0 for hue and range = 1 for saturation and brightness
        # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
        hue_upper = check_boundaries(pixel[0], 10, 0, 1)
        hue_lower = check_boundaries(pixel[0], 10, 0, 0)
        saturation_upper = check_boundaries(pixel[1], 20, 1, 1)
        saturation_lower = check_boundaries(pixel[1], 20, 1, 0)
        value_upper = check_boundaries(pixel[2], 40, 1, 1)
        value_lower = check_boundaries(pixel[2], 40, 1, 0)

        upper =  np.array([hue_upper, saturation_upper, value_upper])
        lower =  np.array([hue_lower, saturation_lower, value_lower])
        print(lower, upper)

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        mask = cv2.inRange(image_hsv,lower,upper)

        # slice the colour
        imask = mask>0
        holds = np.zeros_like(image_src, np.uint8)
        holds[imask] = image_src[imask]
        holds_grey = cv2.cvtColor(holds, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(holds_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contour drawing
        contour_areas = []
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 50:
                contour_areas.append(area) 
        average = np.average(contour_areas)
        contours = [cont for cont in contours if not cv2.contourArea(cont) < average/2]

        cv2.drawContours(holds, contours, -1, (0,0,255), 2)

        cv2.imshow('output', resize_image(holds,width=650))

def main():

    global image_hsv, pixel, image_src

    #OPEN DIALOG FOR READING THE IMAGE FILE
    root = tk.Tk()
    root.withdraw() #HIDE THE TKINTER GUI
    file_path = filedialog.askopenfilename(filetypes = ftypes)
    root.update()
    image_src = cv2.imread(file_path)
    cv2.imshow("BGR",resize_image(image_src,500))

    #CREATE THE HSV FROM THE BGR IMAGE
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)

    #CALLBACK FUNCTION
    cv2.setMouseCallback("BGR", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
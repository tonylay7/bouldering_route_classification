import cv2 as cv
import numpy as np
import sys
import re

def resize_image(image, width=None, height=None, inter=cv.INTER_AREA):
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

    return cv.resize(image, dim, interpolation=inter)

class ImageProcessor:
    img_src = None
    img_name = None
    resized = None
    contours = None

    def __init__(self, src_loc):
        # Initialise variables
        self.img_src = cv.imread(src_loc)
        self.img_name = re.findall(r'[^\/]+(?=\.)',src_loc)[0]
        
    def getColour(self, img_name):
        colour = img_name.split('_')[0]
        if colour == 'green':
            return (36, 25, 25), (70, 255,255)
        elif colour == 'red':
            return (170, 70, 50), (180, 255, 255)
        elif colour == 'purple':
            return (300,)
        elif colour == 'orange':
            return (10, 100, 20), (25, 255, 255)
        else:
            raise ValueError('The file name has a colour that cannot be identified. ('+self.img_name+')')
    def generate_data(self):
        ## convert to hsv
        hsv = cv.cvtColor(self.img_src, cv.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # create mask based on colour
        lower, upper = self.getColour(self.img_name)
        mask = cv.inRange(hsv, lower, upper)

        ## slice the colour
        imask = mask>0
        holds = np.zeros_like(self.img_src, np.uint8)
        holds[imask] = self.img_src[imask]
        holds_grey = cv.cvtColor(holds, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(holds_grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # contour drawing
        contour_areas = []
        for cont in contours:
            area = cv.contourArea(cont)
            if area > 50:
                contour_areas.append(area) 
        average = np.average(contour_areas)
        self.contours = [cont for cont in contours if not cv.contourArea(cont) < average/2]

        cv.drawContours(holds, self.contours, -1, (0,0,255), 2)

        cv.imshow('output', resize_image(holds,width=650))
        
        # De-allocate any associated memory usage 
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
    
def main():
    if len(sys.argv) < 2:
        print("Error not enough arguments.")
    else:
        image_processor = ImageProcessor(sys.argv[1])
        image_processor.generate_data()
    
if __name__ == "__main__":
    main()
 

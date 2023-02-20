import cv2
import numpy as np
import sys
import re

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

class ImageProcessor:
    img_src = None
    img_name = None
    resized = None
    contours = None

    def __init__(self, src_loc):
        # Initialise variables
        self.img_src = cv2.imread(src_loc)
        self.img_name = re.findall(r'[^\/]+(?=\.)',src_loc)[0]
        
    def getColour(self,colour):
        if colour == 'green':
            return (40, 25, 25), (70, 255,255)
        elif colour == 'red':
            return (170, 5, 50), (180, 255, 255)
        elif colour == 'purple':
            return (120,25,40), (170, 255, 255)
        elif colour == 'orange':
            return (10, 50, 20), (25, 255, 255)
        else:
            raise ValueError('The file name has a colour that cannot be identified. ('+self.img_name+')')

    def filterContours(self, contours):
        contour_areas = [cv2.contourArea(cont) for cont in contours]
        contour_areas = [area for area in contour_areas if area>50]
        contour_areas.sort()
        return contour_areas

    def generate_data(self,colour):
        ## convert to hsv
        hsv = cv2.cvtColor(self.img_src, cv2.COLOR_BGR2HSV)

        # create mask based on colour
        lower, upper = self.getColour(colour)
        mask = cv2.inRange(hsv, lower, upper)

        # slice the colour
        imask = mask>0
        holds = np.zeros_like(self.img_src, np.uint8)
        holds[imask] = self.img_src[imask]
        holds_grey = cv2.cvtColor(holds, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(holds_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get only the countours that are holds (so filter out any contours that are too small or large)

        contour_areas = self.filterContours(contours)

        average = np.median(contour_areas)
        self.contours = [cont for cont in contours if not cv2.contourArea(cont) < average/2]

        cv2.drawContours(self.img_src, self.contours, -1, (0,0,255), 2)

        cv2.imshow('output', resize_image(self.img_src,width=650))
        
        # De-allocate any associated memory usage 
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    
def main():
    if len(sys.argv) < 3:
        print("Error not enough arguments.")
    else:
        image_processor = ImageProcessor(sys.argv[1])
        image_processor.generate_data(sys.argv[2])
    
if __name__ == "__main__":
    main()
 

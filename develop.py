import cv2 as cv
import numpy as np
import sys

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
    resized = None
    def __init__(self, src_loc):
        print(src_loc)
        self.img_src = cv.imread(src_loc)
    def generate_data(self):
        ## convert to hsv
        hsv = cv.cvtColor(self.img_src, cv.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # create mask based on colour
        mask = cv.inRange(hsv, (36, 25, 25), (70, 255,255))

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
        median = np.median(contour_areas)
        print(contour_areas)
        filtered_contours = [cont for cont in contours if not cv.contourArea(cont) < median/2]

        cv.drawContours(holds, filtered_contours, -1, (0,0,255), 2)

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
 

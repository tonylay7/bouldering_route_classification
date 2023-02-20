import cv2
import numpy as np
import sys
from sklearn import cluster
from matplotlib import pyplot as plt

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

def preprocess_image(img_path):
    img_src = cv2.imread(img_path)
    img = resize_image(img_src,width=500)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray_blur = cv2.GaussianBlur(imgray,(3,3),0)

    thresh, img_bw_2 = cv2.threshold(imgray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, thresh = cv2.threshold(img_bw_2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 100 and area < 5000:
            filtered_contours.append(contour)

    result_img = img.copy()
    cv2.drawContours(result_img, filtered_contours, -1, (0,0,255), 1)
    cv2.imshow("image",result_img)

        # De-allocate any associated memory usage 
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
   
def main():
    if len(sys.argv) < 2:
        print("Error: Not enough arguments.")
    else:
        img_path = sys.argv[1]
        preprocess_image(img_path)
        
if __name__ == "__main__":
    main()
 
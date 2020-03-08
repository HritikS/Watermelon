import cv2
import numpy as np


def getArea(img, n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur = cv2.blur(img, (5, 5))
    
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 137, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=5) 
    contours, hierarchy = cv2.findContours(
                                   image = thresh, 
                                   mode = cv2.RETR_TREE, 
                                   method = cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    boxes = []
    img_copy = img.copy()
    area = []
    for i in range(n):
        x, y, w, h = cv2.boundingRect(contours[i])
        box = cv2.boxPoints(cv2.minAreaRect(contours[i]))
        box = box.astype('int')
        boxes.append(box)
        cv2.putText(img_copy, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 2)
        area.append(cv2.contourArea(contours[i]))
    
    cv2.imwrite('./output.png', cv2.drawContours(img_copy, contours = boxes, contourIdx = -1, color = (255, 0, 0), thickness = 2))

    print("Image saved!")
    
    return area


if __name__ =="__main__":
    image = cv2.imread('./sampleSemi.png')
    print("Areas: ", getArea(image, 2))
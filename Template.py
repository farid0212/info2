import cv2
import numpy

img = cv2.imread('img/img1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('Tmpl/Tmp12.jpg', 0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.37
loc = numpy.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
cv2.imshow('Result', img)
cv2.waitKey(0)

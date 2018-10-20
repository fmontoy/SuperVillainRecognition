import numpy as np
import cv2 as cv
plate = cv.CascadeClassifier('licenseClassifier.xml')
img = cv.imread('../Dataset/Originales/18-09-2018_17-42-43_KBR291.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plates = plate.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in plates:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
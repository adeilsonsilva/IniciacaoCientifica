import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/matheusm/libfreenect2/examples/protonect/Cascades/cascade.xml')

img = cv2.imread('0009-30-a.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #crop_img = img[y: y + h, x: x + w]

#cv2.imshow('crop_img',crop_img)
cv2.imwrite('face.png', img)
cv2.imshow('imagem',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

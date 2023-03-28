import cv2

img = cv2.imread('img/img1.jpg')
face_cascade = cv2.CascadeClassifier('face.xml')
faces = face_cascade.detectMultiScale(img, scaleFactor=1.5,minNeighbors=1, minSize=(40,40))
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w, y+h),(0, 0, 0),2)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

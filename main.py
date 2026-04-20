import cv2

# Capture frames from the video
video = cv2.VideoCapture('video.avi')

car_cascade = cv2.CascadeClassifier('cars.xml')
while True:
    ret,frames = video.read()

    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    # Detect cars of different sizes
    cars = car_cascade.detectMultiScale(gray,1.1,1)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,100,50),2)
        
    cv2.imshow('Car recognition',frames)
    
    if cv2.waitKey(33) == 27:
        break
cv2.destroyAllWindows()
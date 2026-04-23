import cv2

video = cv2.VideoCapture('video.avi')
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frames = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    
    car_count = len(cars)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 100, 50), 2)
        
    cv2.putText(frames, f'Cars counted: {car_count}', (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        
    cv2.imshow('Car recognition', frames)
    
    if cv2.waitKey(33) == 27:
        break

video.release()
cv2.destroyAllWindows()

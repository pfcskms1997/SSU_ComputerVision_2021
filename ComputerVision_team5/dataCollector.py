import cv2

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        count = count + 1
        name = './images/face_mask' + str(count) + '.jpg'
        print("Creating Images.....")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("WindowFrame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


#faces = facedetect.detectMultiScale(frame, 1.3, 5)
#입력이미지, 검색윈도우 확대 비율, 검출영역으로 선택하기 위한 최소 검출 횟수

#facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#얼굴 검출을 위한 분류기를 불러온다.
#cv2.imwrite(name, frame[y:y+h, x:x+w])
#이미지를 저장하는데 얼굴이 검출된 해당 영역만 저장을 시킨다.
#이미지 이름 및 경로, 저장할 이미지 크기, y, x 순으로 저장한다.


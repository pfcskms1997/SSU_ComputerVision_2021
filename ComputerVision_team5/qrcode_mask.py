import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model
from pyzbar.pyzbar import decode

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold = 0.95
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX
model = load_model('MyTrainingModel.h5') #학습된 마스크 파일

with open('myDataFile.txt') as f: #qr코드 명부 txt 파일을 불러온다.
    myDataList = f.read().splitlines()

def preprocessing(img): #이미지 전처리 정규화
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


def get_className(classNo):
    if classNo == 0:
        return "Mask"
    elif classNo == 1:
        return "No Mask"


while True:
    sucess, imgOrignal = cap.read() # 웹캠에서 읽어온 이미지, 읽어온 것이 성공적이면 success에 1이 들어간다.
    #그리고 img에는 웹캠에서 읽은 이미지 정보가 들어간다.
    faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
    for x,y,w,h in faces:
        # cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
        # cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
        crop_img = imgOrignal[y:y+h,x:x+h] #얼굴만 자른 이미지
        img = cv2.resize(crop_img, (32,32))
        img = preprocessing(img) #이미지 전처리 일반화 함수
        img = img.reshape(1, 32, 32, 1)
        prediction = model.predict(img) #이미지 예측 (학습된 데이터 기반)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(prediction)
        if probabilityValue > threshold: # 임계값보다 값이 크다면 if 문 수행
            if classIndex == 0: # 마스크 착용
                cv2.rectangle(imgOrignal, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.rectangle(imgOrignal, (x,y-40), (x+w, y), (0,255,0), -2)
                cv2.putText(imgOrignal, str(get_className(classIndex)), (x,y-10), font, 0.75, (255,255,255), 1, cv2.LINE_AA)
            elif classIndex == 1: # 마스크 미착용
                cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
                cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
                cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255), 1, cv2.LINE_AA)


    for barcode in decode(imgOrignal):
        myData = barcode.data.decode('utf-8') #그냥 data를 추출하게 되면 b'20160362'이 된다. utf8형식으로 decode해서 없앤다.
        print(myData)

        if myData in myDataList: #qr코드가 리스트 안에 있는 정보인지 아닌지 파악한다.
            print('OK!!')
            myOutput = 'OK!!'
            myColor = (0, 255, 0)
        else:
            print("NOT OK")
            myOutput = 'NOT OK'
            myColor = (0, 0, 255)

        pts = np.array([barcode.polygon], np.int32) #qr코드에 경계선을 만들어주기위한 형식적 절차
        #print("pts no reshape:", pts, pts.shape)
        pts = pts.reshape((-1,1,2))
        #print("pts with reshpe", pts, pts.shape)
        cv2.polylines(imgOrignal, [pts], True, myColor, 5)
        #선 긋는 함수 -> 이미지, 좌표점, 도형 닫힘 유무, 색상, 선 두께
        pts2 = barcode.rect
        #pts2[0], pts[1]은 left, top 좌표에 해당 문자열의 bottom-left를 저 좌표에 위치시킨다.
        cv2.putText(imgOrignal, myOutput, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
        #캡처된 이미지에 data를 텍스트형식으로 넣어준다. 텍스트 입힐 이미지, 문자열, 위치설정, 폰트, 글자크기, 글자 색상, 굵기

    cv2.imshow("Result",imgOrignal)
    k=cv2.waitKey(1)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()














# $ git clone https://github.com/subhamroy021/Facial-Recognition.git
# $ cd Faical_Recognition
# $ mkdir faces

import os
import cv2  #opencv 라이브러리
import keyboard  #키보드 입력 감지 라이브러리
import numpy as np
from os import listdir
from os.path import isfile, join
import warnings

warnings.filterwarnings("ignore", category = np.VisibleDeprecationWarning)

#1.얼굴 사진 찍기
isTrue = False  #유사도 계산이 완료되었는가
isUnlocked = False  #잠금해제에 성공했는가
alpha = 0.25
offset = 6
txt_size = 0.65

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#전체 사진에서 얼굴 부위만 잘라 리턴
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #흑백처리
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  #얼굴 찾기
    if faces == ():  #찾은 얼굴이 없으면 None으로 리턴
        return None
    for(x, y, w, h) in faces:  #얼굴들이 있으면
        cropped_face = img[y: y + h, x: x + w]  #해당 얼굴 크기만큼 cropped_face에 잘라 넣기
        return cropped_face

print("")
print("Colleting..")
print("")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #카메라 실행
count = 0  #저장할 이미지 카운트 변수
countIs = 0

while True:
    ret, frame = cap.read()  #카메라로 부터 사진 1장 얻기
    frame = cv2.flip(frame, +1)  #이미지 좌우반전
    if face_extractor(frame) is not None:  #얼굴 감지 하여 얼굴만 가져오기
        count += 1
        countIs = 0
        text = str(count)
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, txt_size, 2)  #문자열의 크기 알아내기

        face = cv2.resize(face_extractor(frame), (450, 450))  #얼굴 이미지 크기를 500x500으로 조정
        face_cp = face  #face의 복사본 생성
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  #조정된 이미지를 흑백으로 변환
        face_cp = cv2.cvtColor(face_cp, cv2.COLOR_BGR2GRAY)

        file_name_path = 'faces/user' + str(count) + '.jpg'  #faces폴더에 jpg파일로 저장  #ex > faces/user0.jpg   faces/user1.jpg ....
        cv2.imwrite(file_name_path, face)

        #격자 무늬 표시하기
        cv2.line(face, (150, 0), (150, 450), (255, 255, 255), 1)
        cv2.line(face, (300, 0), (300, 450), (255, 255, 255), 1)
        cv2.line(face, (0, 150), (450, 150), (255, 255, 255), 1)
        cv2.line(face, (0, 300), (450, 300), (255, 255, 255), 1)

        #숫자 뒤 검은 박스 나타내기
        cv2.rectangle(face, (20 - offset, 430 + offset), (20 + textSize[0][0] + offset, 430 - textSize[0][1] - offset), (0, 0, 0), -1)
        face = cv2.addWeighted(face, alpha, face_cp, 1 - alpha, 0)  #이미지 합성하기
        cv2.putText(face, text, (20, 430), cv2.FONT_HERSHEY_SIMPLEX, txt_size, (255, 255, 255), 2)  #화면에 얼굴과 현재 저장 개수 표시
        cv2.imshow('Face Cropper', face)

    else:
        countIs += 1
        if countIs == 1:
            print("[INFO] Face not Found")
        else:
            countIs = 2

    if cv2.waitKey(1) == 13 or count == 1000:
       break

    if keyboard.is_pressed('e'):
       break

cap.release()
cv2.destroyAllWindows()

print("")
print("Colleting Samples Complete!!!")
print("")

#2.얼굴 학습
data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]  #faces폴더에 있는 파일 리스트 얻기
Training_Data, Labels = [], []  #데이터와 매칭될 라벨 변수

print("")
print("Training..")
print("")

for i, files in enumerate(onlyfiles):  #파일 개수 만큼 루프
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  #이미지 불러오기

    if images is None:  #이미지 파일이 아니거나 못 읽어 왔다면 무시
        continue

    Training_Data.append(np.asarray(images, dtype = np.uint8))  #Training_Data 리스트에 이미지를 바이트 배열로 추가
    Labels.append(i)  #Labels 리스트엔 카운트 번호 추가
    print("[Traning] user" + str(i + 1) + ".jpg")

if len(Labels) == 0:  #훈련할 데이터가 없다면 종료
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype = np.int32)  #Labels를 32비트 정수로 변환
model = cv2.face.LBPHFaceRecognizer_create()  #모델 생성
model.train(np.asarray(Training_Data), np.asarray(Labels))  #학습 시작

print("")
print("Model Training Complete!!!!!")
print("")

#3.얼굴 인식
def face_detector(img, size = 0.5):
    global isTrue
    global frame_cp
    global confidence

    alpha = 0.75
    offset = 9
    offsetX = 25
    offsetY = 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces == ():
        return img, []

    for (x, y, w, h) in faces:
        # 얼굴 부위 표시하기
        cv2.rectangle(img, (x + offset, y + offset), (x + w - offset, y + h - offset), (255, 255, 255), 1)
        cv2.rectangle(img, (x, y), (x + offsetX, y + offsetY), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + offsetY, y + offsetX), (255, 255, 255), -1)
        cv2.rectangle(img, (x + w, y), (x + w - offsetX, y + offsetY), (255, 255, 255), -1)
        cv2.rectangle(img, (x + w, y), (x + w - offsetY, y + offsetX), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y + h), (x + offsetX, y + h - offsetY), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y + h), (x + offsetY, y + h - offsetX), (255, 255, 255), -1)
        cv2.rectangle(img, (x + w, y + h), (x + w - offsetX, y + h - offsetY), (255, 255, 255), -1)
        cv2.rectangle(img, (x + w, y + h), (x + w - offsetY, y + h - offsetX), (255, 255, 255), -1)


        roi = img[y: y + h, x: x + w]
        roi = cv2.resize(roi, (200, 200))

        img = cv2.addWeighted(img, alpha, frame_cp, 1 - alpha, 0)

        if isTrue:
            cv2.putText(img, display_string, (x, y - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img, roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

print("")
print("Turn on the camera!")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #카메라 열기

while True:
    ret, frame = cap.read()  #카메라로 부터 사진 한장 읽기
    frame = cv2.flip(frame, +1)  #이미지 좌우반전
    ret, frame_cp = cap.read()  #카메라로 부터 사진 한장 읽기
    frame_cp = cv2.flip(frame_cp, +1)  #이미지 좌우반전

    txt_size = 0.8

    image, face = face_detector(frame)  #얼굴 검출 시도

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  #검출된 사진을 흑백으로 변환
        result = model.predict(face)  #학습한 모델로 예측시도

        if result[1] < 500:  #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
            confidence = int(100 * (1 - (result[1]) / 200) + 10)  #유사도 계산
            display_string = str(confidence) + '%'  #유사도 화면에 표시
            isTrue = True

        if confidence > 85 or isUnlocked:  #85% 보다 크면 동일 인물로 간주해 UnLocked!
            isUnlocked = True
            text, color = "Unlocked", (30, 255, 30)

        else:  #85% 이하면 타인.. Locked!!!
            text, color = "Locked", (30, 30, 255)


    except:  #얼굴 검출 안됨
        text, color = "Face Not Found", (240, 240, 240)
        if isUnlocked: text, color = "Unlocked", (30, 255, 30)
        pass



    textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, txt_size, 2)  #문자열의 크기 알아내기

    #화면 중앙에 텍스트 삽입
    cv2.putText(image, text, (int((640 - textSize[0][0]) / 2), 450), cv2.FONT_HERSHEY_SIMPLEX, txt_size, color, 2)
    cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1) == 13:
        break

    if keyboard.is_pressed('e'):
        print("")
        print("Turn off the camera!")
        break

    if keyboard.is_pressed('r'):
        isUnlocked = False


def removeAllFile(file_path):  #faces 폴더 내의 모든 파일 삭제
    if os.path.exists(file_path):
        for file in os.scandir(file_path):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Diretory Not Found"

file_path = 'faces/'
removeAllFile(file_path)

cap.release()
cv2.destroyAllWindows()

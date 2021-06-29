# faceRecognitionLocker

### [이우영 학생의 OPP링크](https://sites.google.com/ydp.hs.kr/202120710)  

```
$ git clone https://github.com/subhamroy021/Facial-Recognition.git
$ python -m pip install keyboard  #keyboard 설치하기
$ python -m pip install opencv-python  #opencv 설치하기
$ mkdir faces  #face폴더 생성하기 
```

### cmd를 열고 해당 폴더 이동 후 아래 명령어를 통해 프로그램 실행
```
$ python test.py
```

### 프로그램 구동 과정
- 얼굴부분을 인식하여 사진을 1000장 확보
- 사진을 faces 폴더에 저장
- 사진을 이용해 사용자 얼굴 학습
- 얼굴인식 락커 작동(e : 프로그램 정지, r : 얼굴이 보이지 않을 때 이 버튼을 누르면 잠금상태로 복귀)
- 프로그램이 정지되면 faces 폴더에 저장되었던 사진은 자동으로 삭제됨

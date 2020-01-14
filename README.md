# SocialCognition: ETRI Social Behavior Recognition Subsystem

ETRI Repository in DeepTask Project for Social Robot

## 라이선스

* 이 소프트웨어의 라이선스는 본 리파지토리에 포함한 [License.md](License.md)와 [Licese_ko.md](License_ko.md)를 따릅니다.

## 변경 사항
* [191204] 개발 언어와 딥러닝 프레임워크 변경, 행동인식 기능 보완
    * C++와 Caffe를 버리고 python과 pytorch를 사용
    * 행동인식 내용 확장
      * **출력값 아이디 의미가 바뀌었으니 아래 "출력 메시지 형식과 토픽"을 꼭 참고해 주세요.**
    * 연령, 성별, 안경 착용 여부, 얼굴 자세는 20프레임에 한번씩 인식하여 결과를 제공
      * *값이 -1인 경우는 인식 결과가 없는 상태를 나타냄*
    * 시선맞춤 여부 인식 결과 제공
      * gaze 필드의 값: 0이면 시선회피(관심無), 1이면 시선맞춤(관심有)
* [190515] OpenPose 기반 소셜행동 인식기 통합 완료
    * 인식 행동의 갯수가 늘어났습니다. (아래 출력 메시지 형식 참조)
    * OpenPose 설치 필요 (설치 방법이 많이 변했으니 반드시 참조 필요함)
    * Intel Realsense와 Nuitrack SDK를 사용하지 않습니다. (Oh Yea!!)
* [181226] 소셜행동 인식 성능 보완, 시선방향 인식 결과 배포
    * 출력 메시지 구조 참조 (social_action과 gaze 필드 참조)
* [181223] 단기적 소셜행동 인식기 연동
    * Realsense D415 카메라와 Nuitrack을 이용한 골격 추적 기능 연계 (KIST의 전폭적 지원!!)
    * 출력 메시지에 행동 아이디 포함 (아래 메시지 구조 참조)
    * 사용상 제약
        * 카메라 앞에 한사람만 존재해야 함

## 기능 소개

ETRI 인식 기능은 etri_recognition_cpp 패키지 안에 구현하고 있습니다. 현재 동작하는 기능은 아래와 같습니다.

* 얼굴 검출
* 얼굴 특징점 추출
* 연령 인식
* 성별 인식
* 안경 착용 여부 인식
* 얼굴 자세 검출
* 얼굴 인식 (*현재는 얼굴 등록 기능을 연계하지 않아 사용 불가*)
* 소셜 행동 인식

## 출력 메시지 형식과 토픽

ETRI 인식 기능의 출력은 아래와 같은 형식의 JSON 문자열로 구성하여 ```recognitionResult``` 토픽으로 배포합니다.

```
   "encoding" : "UTF-8",
   "header" : {                                 // 헤더 정보
      "content" : [ "human_recognition" ],
      "source" : "ETRI",
      "target" : [ "UOA", "UOS" ],
      "timestamp" : 0
   },
   "human_recognition" : [
      {
         "age" : 51.470279693603516,            // 인식기에 의해 추정된 나이값을 실수값으로 기록. -1: 인식 결과 없음
         "face_roi" : {                         // 얼굴 영역 ROI 좌표
            "x1" : 189,
            "x2" : 380,
            "y1" : 249,
            "y2" : 440
         },
         "gender" : 1,                          // 성별 인식 결과 - -1: 결과 없음, 0: 여성, 1: 남성
         "glasses" : false,                     // 안경 착용 여부 - -1: 결과 없음, false: 미착용, true: 착용
         "headpose" : {                         // 얼굴 자세 정보 (Roll, Pitch, Yaw Degree 값)
            "pitch" : -0.66468393802642822,
            "roll" : 8.6534614562988281,
            "yaw" : -7.5606122016906738
         },
         "id" : 0,                              // 추적 아이디 (현재는 기능이 없어 의미없는 값이 저장됨)
         "name" : "",                           // 신원 아이디 (현재는 얼굴인식 기능을 연동하지 않아 의미없는 값이 저장됨)
         "social_action" : -1,                  // 단기적 소셜행동 아이디
                                                // -1: 인식 결과 없음, 
                                                //  0: 손톱 물어뜯기, 1: 입가리기, 2: 화이팅
                                                //  3: 손가락 하트, 4: 오케이
                                                //  5: 팔장끼기, 6: 중립, 7: 귀 후비기
                                                //  8: 턱괴기, 9: 머리긁기
                                                // 10: 악수하기, 11: 엄지척, 12: 코 만지기
                                                // 13: 손 흔들기, 14: 고개숙여 인사
         "gaze": 0                              // 0: 시선회피, 1: 시선맞춤
      }
   ]
}
```

## 설치 방법

### 설치에 필요한 소프트웨어들

#### Video Stream OpenCV
웹캠으로부터 영상을 입수 활용하기 위해 설치합니다. 아래 링크를 따라가서 설치하면 됩니다.

* [video_stream_opencv 패키지](https://github.com/ros-drivers/video_stream_opencv.git)

#### OpenPose
아래 문서의 안내에 따라 설치합니다. 6번 'OpenPose from Other Projects'도 실행해야 합니다.

* [OpenPose 설치 매뉴얼](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#installation)

* **주의할 점**
  * Python 2.7에서 ```import openpose```가 성공해야 합니다. ([여기](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md#compatibility) 참조하세요.)
  * OpenPose 설치 후 PYTHONPATH에 ```/usr/local/python```을 추가해 주세요.

#### OpenCV 3.4 버전 설치
* ROS Kinetic은 OpenCV 3.3을 사용하는데 이 버전으로는 본 패키지 실행 중 오류가 발생합니다.
* OpenCV 3.4 버전 소스를 빌드하여 설치해 주세요.
* 설치 후 PYTHONPATH의 가장 첫머리에 ```/usr/local/python:/usr/local/lib/python2.7/dist-packages```를 추가해 주세요. 이렇게 해야 python이 OpenCV 3.4 런타임을 찾습니다.

### 빌드와 실행

1. 본 패키지와 관련 패키지를 복제합니다.

    ```
    git clone https://github.com/deep-task/ETRI_Recognition.git
    ```

2. ```etri_recognition_py/action_for_Social/models``` 폴더에 검출 인식용 모델 데이터를 저장합니다.
   * 아래 링크에서 다운로드합니다.
     * 링크: https://drive.google.com/drive/folders/1gbpdepw1W5GeSCFQlRge9mFkHr_XDrRx?usp=sharing
     * 필요한 모델 파일들
       * OpenPose의 모델 폴더들
         * face
         * hand
         * pose
         * cameraParametersETRI_AGE.pth.tar
       * ETRI_BODYACTION.pth.tar
       * ETRI_FV.pth.tar
       * ETRI_GENDER.pth.tar
       * ETRI_GLASSES.pth.tar
       * ETRI_HEAD_POSE.pth.tar
       * ETRI_LANDMARK.pth.tar
       * opencv_ssd.caffemodel
       * opencv_ssd.prototxt

3. 다음과 같이 ETRI_Recognition를 구동합니다.

   ```
   roslaunch etri_recognition_py etri_recognition.launch
   ```

#### 연락처
장민수(minsu(at)etri.re.kr)

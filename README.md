# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---
※ 파이썬이 설치되어 있어야 합니다. ※

## 사용 방법
1. 위의 다운로드 버튼을 눌러서 파일을 다운로드하고 압축을 풀어주세요.

![다운로드](./image/1.png)
![다운로드2](./image/2.png)

2. YOLO 웹 사이트에서 YOLOv3 weights파일을 다운받습니다. [YOLO website](https://pjreddie.com/media/files/yolov3.weights).
3. 다운받은 weights 파일을 keras-yolo3 폴더에 넣습니다.
4. keras-yolo3 폴더에서 cmd 창을 실행시키고 아래의 명령어를 입력합니다.

```
python -m pip install keras tensorflow numpy Pillow matplotlib
```
※ 위 코드에서 에러가 발생하면 파이썬 환경변수 설정이 되어있나 확인해주세요 ※

```
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
5. 변환 작업이 끝나면 아래 명령어를 입력해 YOLO를 사용합니다.
```
$python yolo_video.py --image
    Using TensorFlow backend.
    Image detection mode
    Ignoring remaining command line arguments: ./path2your_video,
    ~~~
    ~~~
    ~~~
    Instructions for updating:
    Colocations handled automatically by placer.
    model_data/yolo.h5 model, anchors, and classes loaded.
    Input image filename:test/dog.jpg
``` 
                     ↑↑↑↑ test/dog.jpg 말고 다른 이미지 파일도 사용해보세요

결과

![결과](./image/3.jpg)

### YOLO 사용법
Use --help to see usage of yolo_video.py:
yolo_video.py 사용법
```
usage: yolo_video.py [-h] [--model ] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        동영상 인풋 경로
  --output       동영상 아웃풋 경로

선택 인자:
  -h, --help         show this help message and exit
  --model MODEL      모델 weight 파일 경로, 기본 경로는 model_data/yolo.h5
  --anchors ANCHORS  anchor definitions 경로, 기본 경로는
                     model_data/yolo_anchors.txt
  --classes CLASSES  class definitions 경로, 기본 경로는
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  사용할 GPU의 개수, 기본 1
  --image            이미지 인식 모드, 다른 인자들을 무시합니다.
```



Tiny YOLOv3의 경우 비슷한 방법으로 모델 경로와 앵커 경로를 '--model model_file'와 '--anchors anchor_file'로 지정하면됩니다.

4. 다중GPU 사용방법 :`--gpu_num N '을 사용하여 N 개의 GPU를 사용합니다. 이것은 [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model)에 상속 됩니다.

## 학습시키기

1. 자신의 주석 파일 및 클래스 이름 파일을 생성합니다.  
    하나의 이미지에 대한 하나의 행;  
    행 형식: `image_file_path box1 box2 ... boxN`;  
    박스 형식: `x_min,y_min,x_max,y_max,class_id` (공백 없음).  
    VOC 데이터셋의 경우 `python voc_annotation.py`  
    예제:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. `python convert.py -w yolov3.cfg yolov3.weights model_data / yolo_weights.h5`를 실행했는지 확인하세요.
    model_data / yolo_weights.h5 파일은 사전 가중치를 로드하는 데 사용됩니다.

3. train.py를 수정하고 학습을 시작하세요.
    `python train.py`
    yolo_video.py를 사용할 때 훈련 된 가중치 또는 체크 포인트 가중치를`--model model_file` 명령 행 옵션으로 사용하세요.
    클래스 경로 나 앵커 경로는`--classes class_file`와`--anchors anchor_file`으로 수정해야합니다.


YOLOv3에 원래의 가중치를 사용하려면 다음 방법대로 진행하세요.
    1.`wget https : // pjreddie.com / media / files / darknet53.conv.74`
    2. darknet53.weights로 이름을 변경하세요.
    3.`python convert.py -w darknet53.cfg darknet53.weights model_data / darknet53_weights.h5`
    4. train.py에서 model_data / darknet53_weights.h5를 사용하세요.

---

## 알고 있어야 할 몇 가지 문제

1. 테스트 환경
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. 기본 앵커가 사용됩니다. 자신 만의 앵커를 사용한다면 아마도 약간의 변경이 필요할 것입니다.

3. 추론 결과는 Darknet과 완전히 같지는 않지만 차이는 작다.

4. 속도가 다크 넷보다 느립니다. PIL을 opencv로 바꾸면 조금 도움이 될 것입니다.

5. 훈련의 첫 번째 단계에서는 항상 예비 가중치를로드하고 레이어를 고정시킵니다. 또는 Darknet 교육을 시도하십시오. 불일치 경고가 있으면 괜찮습니다.

6. 교육 전략은 참고 용입니다. 데이터 세트와 목표에 따라 조정하십시오. 필요한 경우 추가 전략을 추가하십시오.

7. 얼어 붙은 레이어로 교육 과정을 가속화하기 위해 train_bottleneck.py를 사용할 수 있습니다. 동결 된 모델의 병목 현상을 먼저 계산 한 다음 마지막 레이어 만 교육합니다. 따라서 합리적인 시간에 CPU 교육이 가능합니다. 병목 현상에 대한 자세한 내용은 [this] (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)를 참조하십시오.
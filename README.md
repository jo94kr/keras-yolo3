# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## 사용 방법

1. YOLO 웹 사이트에서 YOLOv3 weights파일을 다운받습니다. [YOLO website](https://pjreddie.com/media/files/yolov3.weights).
2. 다운받은 weights 파일을 keras-yolo3 폴더에 넣습니다.
3. keras-yolo3 폴더에서 cmd 창을 실행시키고 아래의 명령어를 입력합니다.

```
pip install keras tensorflow numpy
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
4. 변환 작업이 끝나면 아래 명령어를 입력해 YOLO를 사용합니다.
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
---


For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.


4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## 학습시키기

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

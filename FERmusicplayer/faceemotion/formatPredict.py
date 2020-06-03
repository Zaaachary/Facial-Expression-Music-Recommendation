"""
Author: LZF Zachary
Website: zrawberry.com
Filename: Preprocess.py
"""
import os
import cv2
import numpy as np

from .Network import *

# print(os.getcwd()) # D:\CODE\FERmusicplayer
CASC_PATH = './faceemotion/nnSource/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

# 加载模型
model1 = FerModel()
if not os.path.exists(checkpoint_save_path + str(1) + '.index'):
    
    print("请先训练模型并生成checkpoint！")
else:
    model1.load_weights(checkpoint_save_path + str(1))


def format_image(image):
    """
    格式化图片并框出人脸
    :param image:
    :return: 人脸图 以及 原图上的人脸坐标
    """
    if len(image.shape) > 2 and image.shape[2] in (3, 4):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor = max_are_face
    image = image[face_coor[1]:(
        face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    return image, face_coor


def PIL_detect(image):
    # input is a PIL image ouput find face
    # 模型
    model = model1
    img = np.array(image)
    detected_face, face_coor = format_image(img)
    if face_coor is not None:
        [x, y, w, h] = face_coor
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = Image.fromarray(img).convert('RGB')
    face.save('media/pictures/output.png', format='png')
    if detected_face is not None:   # np
        # print(detected_face.shape) # numpy.ndarray  48,48  检测
        cv2.imwrite('media/pictures/detected_face.png', detected_face)
        x_predict = image2numpy(detected_face)
        result = model.predict(x_predict)
        # print(result)
        pred = int(tf.argmax(result, axis=1))  # 输出最大的概率
        print("检测到人脸，预测值为{}，是{}表情。".format(pred, EMOTIONS[pred]))
        return pred, result[0].tolist()
    else:
        return -1, None


def camera_detect(fermodel=1):
    if not os.path.exists(checkpoint_save_path + str(fermodel) + '.index'):
        print("请先训练模型并生成checkpoint！")
        # return None

    if fermodel == 1:
        model = FerModel()
        model.load_weights(checkpoint_save_path+str(1))
    else:
        model = FerModel2()
        model.load_weights(checkpoint_save_path+str(2))
    video_captor = cv2.VideoCapture(0)

    while True:
        ret, frame = video_captor.read()
        detected_face, face_coor = format_image(frame)
        if face_coor is not None:
            [x, y, w, h] = face_coor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if cv2.waitKey(2) & 0xFF == ord(' '):
            if detected_face is not None:
                cv2.imwrite('./Source/last_camera_input.jpg', detected_face)
                x_predict = image2numpy(detected_face)
                result = model.predict(x_predict)
                print(result)
                pred = int(tf.argmax(result, axis=1))  # 输出最大的概率
                print("检测到人脸，预测值为{}，是{}表情。".format(pred, EMOTIONS[pred]))

        cv2.imshow("Camera Face Emotion Detect ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("结束".center(60, '='))
            break


if __name__ == '__main__':
    camera_detect(2)
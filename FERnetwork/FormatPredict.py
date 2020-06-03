"""
Author: LZF Zachary
Website: zrawberry.com
Filename: FormatPredict.py
"""
import cv2
import numpy as np
from PIL import Image

from Network import *

CASC_PATH = './Source/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
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


def pillow_image(image):
    # npimg = np.array(image.convert('L'))
    npimg = np.array(image)
    # img.show()
    detected_face, face_coor = format_image(npimg)


def camera_dect(fermodel=2):
    if not os.path.exists(checkpoint_save_path + str(fermodel) + '.index'):
        print("请先训练模型并生成checkpoint！")
        return None
    else:
        print("已找到checkpoint")

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
                print(detected_face.shape)
                x_predict = image2numpy(detected_face)
                print(x_predict.shape)
                result = model.predict(x_predict)
                print(result)
                pred = int(tf.argmax(result, axis=1))  # 输出最大的概率
                print("检测到人脸，预测值为{}，是{}表情。".format(pred, EMOTIONS[pred]))

        # if result is not None:
        #     for index, emotion in enumerate(EMOTIONS):
        #         cv2.putText(frame, emotion, (10, index * 20 + 20),
        #                     cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
        #         cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
        #                       (255, 0, 0), -1)
        #         emoji_face = feelings_faces[np.argmax(result[0])]
        #
        #     for c in range(0, 3):
        #         frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
        #             (emoji_face[:, :, 3] / 255.0) + frame[200:320,
        #                                                   10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

        cv2.imshow("Camera Face Emotion Detect ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("结束".center(60, '='))
            break


if __name__ == '__main__':
    camera_dect(2)
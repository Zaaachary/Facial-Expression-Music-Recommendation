"""
Author: LZF Zachary
Website: zrawberry.com
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from PIL import Image
from matplotlib import pyplot as plt

from Utils import load_data

# [0'愤怒', 1'恶心', 2'恐惧', 3'快乐', 4'悲伤', 5'惊讶', 6'平静'] -> [0'愤怒', 1'快乐', 2'悲伤', 3'惊讶', 4'平静']
EMOTIONS = ['愤怒', '快乐', '悲伤', '惊讶', '平静']
checkpoint_save_path = "./checkpoint/fermodel.ckpt"


class FerModel(Model):
    def __init__(self):
        super(FerModel, self).__init__()
        self.c1 = Conv2D(filters=32, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=1)

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=1)

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                         activation='relu')

        # self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
        #                  activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(64, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(5, activation='softmax')    # 去掉恐惧 恶心 7->5

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 48, 48, 1])
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        # x = self.c4(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


class FerModel2(Model):
    def __init__(self):
        super(FerModel2, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(5, 5))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.flatten = Flatten()
        self.f1 = Dense(384, activation='relu')
        self.f2 = Dense(192, activation='relu')
        self.f3 = Dense(5, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 48, 48, 1])
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


def train_network(fermodel=1, epochs=10, ckpt=False):
    """
    :param ckpt: 是否断点续训
    :return:
    """
    X_train, Y_train = load_data('Train')
    X_test, Y_test = load_data('Test')
    X_train, Y_train = X_train / 255.0, Y_train / 1.0
    X_test, Y_test = X_test / 255.0, Y_test / 1.0

    if fermodel == 1:
        model = FerModel()
    else:
        model = FerModel2()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    if ckpt:
        # checkpoint 断点续训
        if os.path.exists(checkpoint_save_path + str(fermodel) + '.index'):
            print("加载 checkpoint 继续训练".center(60, '='))
            model.load_weights(checkpoint_save_path+str(fermodel))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path+str(fermodel),
                                                         save_weights_only=True,
                                                         save_best_only=True)
        history = model.fit(X_train, Y_train, batch_size=64, epochs=epochs,
                            validation_data=(X_test, Y_test), validation_freq=1,
                            callbacks=[cp_callback])
        print("已经保存 checkpoint".center(60, '='))
    else:
        history = model.fit(X_train, Y_train, batch_size=128, epochs=epochs,
                            validation_data=(X_test, Y_test), validation_freq=1, )
        print("训练完成".center(60, '='))

    input("输入 Enter 查看 Summary")
    model.summary()
    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def image2numpy(image):
    # 将图片铺平并归一化
    return np.asarray(image).reshape(-1, 2304) / 255.0
    # return np.asarray(image).reshape(-1, 2304) / 1.0


def predict(x_predict):
    model = FerModel()
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)  # 输出最大的概率

    return pred


if __name__ == "__main__":
    # 训练模型
    train_network(fermodel=1, epochs=20, ckpt=True)

    # 预测last_camera_input.jpg
    # image_path = './Source/last_camera_input.jpg'
    # img = Image.open(image_path).resize((48, 48), Image.ANTIALIAS)
    # img_arr = image2numpy(img.convert('L'))
    # x_predict = img_arr
    # pred = int(predict(x_predict))
    # print(type(pred))
    # print("预测值为{}，是{}表情。".format(pred, EMOTIONS[pred]))

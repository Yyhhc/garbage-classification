import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from time import *


def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=6):
    # 微调的过程中不需要进行归一化的处理
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    # 备选方案是resnet
    # model = keras.Sequential(name='AlexNet')
    # model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE))
    # model.add(layers.Conv2D(96, (11, 11), strides=(2, 2),
    #                         padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(
    #     layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(
    #     layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(class_num, activation='softmax'))
    # model.summary()
    # # 模型训练
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # return model


# 展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results_resnet50_epoch30.png', dpi=100)
    # plt.show()


def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load("F:/datas/tmp/data/tttt/trash_jpg", 224, 224, 16)
    print(class_names)
    model = model_load(class_num=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    model.save("models/resnet50_245_epoch30.h5")
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)
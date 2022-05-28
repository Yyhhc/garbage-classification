import tensorflow as tf
import matplotlib.pyplot as plt
from time import *



# 数据加载，将会按照8比2的比例分割数据集，其中8份作为训练集，2份作为测试集‘
# todo 数据集下载地址：https://download.csdn.net/download/ECHOSON/19713816
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,  # 划分比例
        subset="training",  # 训练集
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,  # 划分比例
        subset="validation",  # 验证集
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names  # 获取数据集的类名
    return train_ds, val_ds, class_names  # 返回训练集、验证集和类名


# 模型加载
def model_load(IMG_SHAPE=(224, 224, 3), class_num=245):
    # 通过keras构建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),  # 归一化，将像素值处理成0到1之间的值
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
        tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
        # Add another convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
        tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
        tf.keras.layers.Flatten(),  # 将二维的特征图拉直
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),  # 128个神经元的全连接层
        tf.keras.layers.Dense(class_num, activation='softmax')  # 输出层，对应数据集具体的类别数目
    ])
    model.summary()  # 输出模型信息
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])  # 编译模型，指定模型的优化器是adam，模型使用的损失函数的交叉熵损失函数
    return model  # 返回模型


# 展示训练过程的曲线，即你在results目录下看到的折线图是通过这里的函数绘制的
def show_loss_acc(history):
    # 从history中获取准确率信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # 从history中获取loss信息
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # 绘制上方的准确率折线图
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    # 绘制下方的损失折线图
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_cnn.png', dpi=100)


# cnn模型训练的主流程
def train(epochs):
    begin_time = time()  # 记录开始时间
    train_ds, val_ds, class_names = data_load("F:/datas/tmp/data/tttt/trash_jpg", 224, 224,
                                              16)  # todo 加载数据 修改为你自己的数据集路径
    print(class_names)  # 输出类名
    model = model_load(class_num=len(class_names))  # 加载模型
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)  # 开始训练
    model.save("models/cnn_245_epoch30.h5")  # todo 训练之后的模型保存在models目录下
    end_time = time()  # 结束时间
    run_time = end_time - begin_time  # 记录时间差
    print('该循环程序运行时间：', run_time, "s")  # 输出时间
    show_loss_acc(history)  # 保存折线图


if __name__ == '__main__':
    train(epochs=30)  # todo epoch可以设置为你想要的轮数

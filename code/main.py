import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from time import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.models.load_model("models/cnn_245_epoch30.h5")



#
# # train_datset = ImageDataGenerator(rescale=1./255)
# # valid_datset = ImageDataGenerator(rescale=1./255)
# data_gen = ImageDataGenerator()
# train_generator = data_gen.flow_from_directory(
#     r'C:\Users\HTQ\Desktop\Class\trash_classification_tf2.3-master\dataset\train',
#     target_size=(28,28),
#     batch_size=16)
# test_generator = data_gen.flow_from_directory(
#     r'C:\Users\HTQ\Desktop\Class\trash_classification_tf2.3-master\dataset\test',
#     target_size=(28,28),
#     batch_size=16)
# print(train_generator.n)
# print(train_generator.labels)
# print(train_generator.num_classes)
# print(train_generator.class_indices)
# print(dir(train_generator))
#
# #构建模型
# #1. 设置层
# model = keras.models.Sequential([
#     # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(128,128,3)),  # 归一化，将像素值处理成0到1之间的值
#     # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
#     # tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
#     #
#     # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
#     # tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
#     #
#     # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
#     # tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
#
#     keras.layers.Flatten(input_shape=(28,28,3)),   #将图像从28*28的二维数组转化为一维数组(28*28=784)
#     keras.layers.Dense(512, activation='relu'),  #
#     keras.layers.Dense(18,activation='sigmoid')
# ])
# model.summary()
# #2. 编译模型
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# #训练模型
# model.fit(train_generator, epochs=15, verbose=1, validation_data=test_generator, validation_steps=8)
# #model.fit(train_generator, verbose=1, validation_data=test_generator)
# #model.fit_generator(train_generator, epochs=15, verbose=1, validation_data=test_generator, validation_steps=8)
# #model.fit_generator(train_generator, steps_per_epoch=16, validation_data=test_generator, validation_steps=8)
# #评估模型
# test_loss, test_acc =model.evaluate_generator(test_generator, steps=24)
#
# print('Test loss:', test_loss)
# print('Test accuracy:', test_acc)
#
#
# #预测模型
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_image)
# print(predictions[0])
# print(np.argmax(predictions[0]))


print('model end')





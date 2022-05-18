import tensorflow as tf
import os
import numpy as np
from PIL import Image
import shutil
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

# 可以显示中文
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

# 垃圾类名
class_names = ['其他垃圾_PE塑料袋', '其他垃圾_U型回形针', '其他垃圾_一次性杯子', '其他垃圾_一次性棉签', '其他垃圾_串串竹签', '其他垃圾_便利贴', '其他垃圾_创可贴',
               '其他垃圾_卫生纸', '其他垃圾_厨房手套', '其他垃圾_厨房抹布', '其他垃圾_口罩', '其他垃圾_唱片', '其他垃圾_图钉', '其他垃圾_大龙虾头',
               '其他垃圾_奶茶杯', '其他垃圾_干燥剂', '其他垃圾_彩票', '其他垃圾_打泡网', '其他垃圾_打火机', '其他垃圾_搓澡巾', '其他垃圾_果壳', '其他垃圾_毛巾',
               '其他垃圾_涂改带', '其他垃圾_湿纸巾', '其他垃圾_烟蒂', '其他垃圾_牙刷', '其他垃圾_电影票', '其他垃圾_电蚊香', '其他垃圾_百洁布', '其他垃圾_眼镜',
               '其他垃圾_眼镜布', '其他垃圾_空调滤芯', '其他垃圾_笔', '其他垃圾_胶带', '其他垃圾_胶水废包装', '其他垃圾_苍蝇拍', '其他垃圾_茶壶碎片',
               '其他垃圾_草帽', '其他垃圾_菜板', '其他垃圾_车票', '其他垃圾_酒精棉', '其他垃圾_防霉防蛀片', '其他垃圾_除湿袋', '其他垃圾_餐巾纸',
               '其他垃圾_餐盒', '其他垃圾_验孕棒', '其他垃圾_鸡毛掸', '厨余垃圾_八宝粥', '厨余垃圾_冰激凌', '厨余垃圾_冰糖葫芦', '厨余垃圾_咖啡',
               '厨余垃圾_圣女果', '厨余垃圾_地瓜', '厨余垃圾_坚果', '厨余垃圾_壳', '厨余垃圾_巧克力', '厨余垃圾_果冻', '厨余垃圾_果皮', '厨余垃圾_核桃',
               '厨余垃圾_梨', '厨余垃圾_橙子', '厨余垃圾_残渣剩饭', '厨余垃圾_水果', '厨余垃圾_泡菜', '厨余垃圾_火腿', '厨余垃圾_火龙果', '厨余垃圾_烤鸡',
               '厨余垃圾_瓜子', '厨余垃圾_甘蔗', '厨余垃圾_番茄', '厨余垃圾_秸秆杯', '厨余垃圾_秸秆碗', '厨余垃圾_粉条', '厨余垃圾_肉类', '厨余垃圾_肠',
               '厨余垃圾_苹果', '厨余垃圾_茶叶', '厨余垃圾_草莓', '厨余垃圾_菠萝', '厨余垃圾_菠萝蜜', '厨余垃圾_萝卜', '厨余垃圾_蒜', '厨余垃圾_蔬菜',
               '厨余垃圾_薯条', '厨余垃圾_薯片', '厨余垃圾_蘑菇', '厨余垃圾_蛋', '厨余垃圾_蛋挞', '厨余垃圾_蛋糕', '厨余垃圾_豆', '厨余垃圾_豆腐',
               '厨余垃圾_辣椒', '厨余垃圾_面包', '厨余垃圾_饼干', '厨余垃圾_鸡翅', '可回收物_不锈钢制品', '可回收物_乒乓球拍', '可回收物_书', '可回收物_体重秤',
               '可回收物_保温杯', '可回收物_保鲜膜内芯', '可回收物_信封', '可回收物_充电头', '可回收物_充电宝', '可回收物_充电牙刷', '可回收物_充电线',
               '可回收物_凳子', '可回收物_刀', '可回收物_包', '可回收物_单车', '可回收物_卡', '可回收物_台灯', '可回收物_吊牌', '可回收物_吹风机',
               '可回收物_呼啦圈', '可回收物_地球仪', '可回收物_地铁票', '可回收物_垫子', '可回收物_塑料制品', '可回收物_太阳能热水器', '可回收物_奶粉桶',
               '可回收物_尺子', '可回收物_尼龙绳', '可回收物_布制品', '可回收物_帽子', '可回收物_手机', '可回收物_手电筒', '可回收物_手表', '可回收物_手链',
               '可回收物_打包绳', '可回收物_打印机', '可回收物_打气筒', '可回收物_扫地机器人', '可回收物_护肤品空瓶', '可回收物_拉杆箱', '可回收物_拖鞋',
               '可回收物_插线板', '可回收物_搓衣板', '可回收物_收音机', '可回收物_放大镜', '可回收物_日历', '可回收物_暖宝宝', '可回收物_望远镜',
               '可回收物_木制切菜板', '可回收物_木桶', '可回收物_木棍', '可回收物_木质梳子', '可回收物_木质锅铲', '可回收物_木雕', '可回收物_枕头',
               '可回收物_果冻杯', '可回收物_桌子', '可回收物_棋子', '可回收物_模具', '可回收物_毯子', '可回收物_水壶', '可回收物_水杯', '可回收物_沙发',
               '可回收物_泡沫板', '可回收物_灭火器', '可回收物_灯罩', '可回收物_烟灰缸', '可回收物_热水瓶', '可回收物_燃气灶', '可回收物_燃气瓶',
               '可回收物_玩具', '可回收物_玻璃制品', '可回收物_玻璃器皿', '可回收物_玻璃壶', '可回收物_玻璃球', '可回收物_瑜伽球', '可回收物_电动剃须刀',
               '可回收物_电动卷发棒', '可回收物_电子秤', '可回收物_电熨斗', '可回收物_电磁炉', '可回收物_电脑屏幕', '可回收物_电视机', '可回收物_电话',
               '可回收物_电路板', '可回收物_电风扇', '可回收物_电饭煲', '可回收物_登机牌', '可回收物_盒子', '可回收物_盖子', '可回收物_盘子', '可回收物_碗',
               '可回收物_磁铁', '可回收物_空气净化器', '可回收物_空气加湿器', '可回收物_笼子', '可回收物_箱子', '可回收物_纸制品', '可回收物_纸牌',
               '可回收物_罐子', '可回收物_网卡', '可回收物_耳套', '可回收物_耳机', '可回收物_衣架', '可回收物_袋子', '可回收物_袜子', '可回收物_裙子',
               '可回收物_裤子', '可回收物_计算器', '可回收物_订书机', '可回收物_话筒', '可回收物_豆浆机', '可回收物_路由器', '可回收物_轮胎', '可回收物_过滤网',
               '可回收物_遥控器', '可回收物_量杯', '可回收物_金属制品', '可回收物_钉子', '可回收物_钥匙', '可回收物_铁丝球', '可回收物_铅球',
               '可回收物_铝制用品', '可回收物_锅', '可回收物_锅盖', '可回收物_键盘', '可回收物_镊子', '可回收物_闹铃', '可回收物_雨伞', '可回收物_鞋',
               '可回收物_音响', '可回收物_餐具', '可回收物_餐垫', '可回收物_饰品', '可回收物_鱼缸', '可回收物_鼠标', '有害垃圾_指甲油', '有害垃圾_杀虫剂',
               '有害垃圾_温度计', '有害垃圾_灯', '有害垃圾_电池', '有害垃圾_电池板', '有害垃圾_纽扣电池', '有害垃圾_胶水', '有害垃圾_药品包装', '有害垃圾_药片',
               '有害垃圾_药瓶', '有害垃圾_药膏', '有害垃圾_蓄电池', '有害垃圾_血压计']


# 数据加载，按照8:2的比例加载垃圾数据
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
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


# 测试mobilenet的准确率
def test_mobilenet():
    train_ds, val_ds, class_names = data_load("F:/datas/tmp/data/tttt/trash_jpg", 224, 224, 4)  # todo 修改为你的数据集的位置
    print(class_names)
    model = tf.keras.models.load_model("models/mobilenet_245_epoch30.h5")  # todo 修改为训练好的mobilenet模型位置
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    print('Mobilenet test accuracy :', accuracy)


# 测试cnn模型的准确率
def test_cnn():
    train_ds, val_ds, class_names = data_load("F:/datas/tmp/data/tttt/trash_jpg", 224, 224, 4)  # todo 修改为你的数据集的位置
    model = tf.keras.models.load_model("models/cnn_245_epoch30.h5")  # todo 修改为训练好的cnn模型位置
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    print('CNN test accuracy :', accuracy)


# 绘制热力图，按照四个大类绘制热力图
# 注：绘制热力图这段的逻辑，因为涉及的数据比较多，为了防止出错，还是先保存为了pkl文件
def draw_heatmap(folder_name):
    # 遍历文件夹返回数目
    trash_names = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']
    real_label = []
    pre_label = []
    images_path = []
    folders = os.listdir(folder_name)

    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        for img in images:
            xxx = folder.split("_")[0]
            x_idx = trash_names.index(xxx)
            img_path = os.path.join(folder_path, img)
            real_label.append(x_idx)
            images_path.append(img_path)

    model = tf.keras.models.load_model("models/mobilenet_245_epoch30.h5")
    for ii, i_path in enumerate(images_path):
        print("{}/{}".format(ii, len(images_path) - 1))
        shutil.copy(i_path, "images/t1.jpg")
        src_i = cv2.imread("images/t1.jpg")
        src_r = cv2.resize(src_i, (224, 224))
        cv2.imwrite("images/t2.jpg", src_r)
        img = Image.open("images/t2.jpg")
        img = np.asarray(img)
        outputs = model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = class_names[result_index]
        names = result.split("_")
        xxx = names[0]
        x_idx = trash_names.index(xxx)
        pre_label.append(x_idx)

    print(len("pre:{}".format(len(pre_label))))
    print(len("real:{}".format(len(real_label))))
    print(pre_label)
    print(real_label)
    # 先保存为pickle文件
    a_dict = {}
    a_dict["pre_label"] = pre_label
    a_dict["real_label"] = real_label
    file = open('results/pickle_result.pickle', 'wb')
    pickle.dump(a_dict, file)
    file.close()


# 加载pkl文件
def load_pickle(filename="pickle_result.pickle"):
    f = open(filename, 'rb')  # 注意此处model是rb
    s = f.read()
    data = pickle.loads(s)
    # print(data)
    pre_label = data['pre_label']
    real_label = data['real_label']
    print(len(pre_label))
    print(len(real_label))
    heatmap = np.zeros((4, 4))
    for r, p in zip(real_label, pre_label):
        heatmap[r][p] = heatmap[r][p] + 1
    # heatmap = preprocessing.scale(heatmap)
    print(heatmap)
    result = []
    for row in heatmap:
        print(row)
        row = row / np.sum(row)
        result.append(row)
    return np.array(result)


# 得到热力图
def get_heat_map(array_numpy=np.random.rand(16).reshape(4, 4)):
    trash_names = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']
    x = array_numpy
    plt.xticks(np.arange(len(trash_names)), trash_names)
    plt.yticks(np.arange(len(trash_names)), trash_names)
    plt.imshow(x, cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.title('classification heatmap')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test_cnn()
    test_mobilenet()

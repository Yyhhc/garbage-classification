import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# 查看图片数量
def read_flower_data(folder_name):
    folders = os.listdir(folder_name)
    flower_names = []
    flower_nums = []
    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        images_num = len(images)
        print("{}:{}".format(folder, images_num))
        flower_names.append(folder)
        flower_nums.append(images_num)

    return flower_names, flower_nums


# 绘制柱状图
def show_bar(x, y, data_length):
    # 绘图
    plt.barh(range(data_length), y, align='center', color='steelblue', alpha=0.8)
    # 添加轴标签
    plt.xlabel('num')
    # 添加标题
    plt.title('Num of trash')
    # 添加刻度标签
    plt.yticks(range(data_length), x)
    # 设置Y轴的刻度范围
    # plt.xlim([32, 47])
    # 为每个条形图添加数值标签
    for x, y in enumerate(y):
        plt.text(y + 0.1, x, '%s' % y, va='center')
    # 显示图形
    plt.show()


def read_trash_data(folder_name):
    folders = os.listdir(folder_name)
    trash_names = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']
    flower_nums = [0 for i in trash_names]
    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        images_num = len(images)
        xxx = folder.split("_")[0]
        num_idx = trash_names.index(xxx)
        flower_nums[num_idx] = flower_nums[num_idx] + images_num

    return trash_names, flower_nums


if __name__ == '__main__':
    # 通用
    # x, y = read_flower_data('F:/datas/tmp/data/tttt/trash_jpg')
    # data_length = len(x)
    # show_bar(x, y, data_length)
    # 垃圾数据集分类
    x, y = read_trash_data('F:/datas/tmp/data/tttt/trash_jpg')
    data_length = len(x)
    show_bar(x, y, data_length)

from struct import unpack
import os
import warnings
warnings.filterwarnings("error", category=UserWarning)

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break

# images = []
# bads = []


def get_result(images_list):
    good_images = []
    bad_images = []
    for img in images_list:
        image = JPEG(img)
        try:
            image.decode()
            good_images.append(img)
        except:
            bad_images.append(img)



# train_folder = "F:/datas/垃圾数据集/trash"
# for subfolder in os.listdir(train_folder):
#     print(subfolder)
#     subfolder_path = os.path.join(train_folder, subfolder)
#     for img_name in os.listdir(subfolder_path):
#         img_path = os.path.join(subfolder_path, img_name)
#         images.append(img_path)

# 清理图片1
# for img in images:
#     image = JPEG(img)
#     try:
#         image.decode()
#     except:
#         bads.append(img)

# 清理图片2
# for img in images:
#     xxx = img.split(".")[-1]
#     try:
#         img = Image.open(img)
#     except:
#         bads.append(img)
#         print('corrupt img', img)

# for img in images:
#     imgx = Image.open(img)
#     imgx_l = len(imgx.split())
#     if imgx_l != 3:
#         print(img)

for name in bads:
    print(name)
    os.remove(name)
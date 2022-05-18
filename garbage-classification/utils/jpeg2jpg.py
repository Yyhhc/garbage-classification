import os
from PIL import Image

train_folder = "F:/datas/tmp/data/tttt/trash"
target_folder = "F:/datas/tmp/data/tttt/trash_jpg"

for subfolder in os.listdir(train_folder):
    subfolder_path = os.path.join(train_folder, subfolder)
    target_folder_path = os.path.join(target_folder, subfolder)
    if os.path.isdir(target_folder_path) == False:
        os.mkdir(target_folder_path)

    for img_name in os.listdir(subfolder_path):

        img_path = os.path.join(subfolder_path, img_name)
        img_new_name = img_name.split(".")[0] + '.jpg'
        img_new_path = os.path.join(target_folder_path, img_new_name)
        src_img = Image.open(img_path)
        src_img.save(img_new_path, quality=95)
        print("{} saved".format(img_new_name))
from struct import unpack
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


def get_result(images_list):
    good_images = []
    bad_images = []
    l = len(images_list)
    for i, img in enumerate(images_list):
        print("{}/{}".format(i+1, l))
        image = JPEG(img)
        try:
            image.decode()
            good_images.append(img)
        except:
            bad_images.append(img)
    return good_images, bad_images

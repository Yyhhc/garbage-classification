import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('Fruit recognition')
        self.model = tf.keras.models.load_model("models/mobilenet_245_epoch30.h5")
        self.to_predict_name = "images/guopi.jpg"
        self.class_names = ['Apple', 'Banana', 'Oranges', 'Apples', 'Banana', 'Oranges']
        self.fff_name = ['Fresh', 'Fresh', 'Fresh', 'Rotten', 'Rotten', 'Rotten']
        self.resize(900, 500)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('Arial', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("Sample")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = h/400
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" Upload image ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" Start to recognize ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' Fruit type ')
        self.result = QLabel("Waiting")
        label_result.setFont(QFont('Arial', 16))
        self.result.setFont(QFont('Arial', 24))

        label_result_f = QLabel(' Freshness ')
        self.result_f = QLabel("Waiting")

        self.label_info = QTextEdit()
        self.label_info.setFont(QFont('Arial', 8))
        # self.label_info.setLineWidth(100)

        label_result_f.setFont(QFont('Arial', 16))
        self.result_f.setFont(QFont('Arial', 24))

        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.label_info)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('Welcome')
        about_title.setFont(QFont('Arial', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/logo.png'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()
        label_super.setFont(QFont('Arial', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        # git_img = QMovie('images/')
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, 'Main')
        self.addTab(about_widget, 'About')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.jpg *.png *jpeg)')
        # print(openfile_name)
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmpx.jpg"
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = h / 400
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))

    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        # gray_img = img.convert('L')
        # img_torch = self.transform(gray_img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        # print(outputs)
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        result_f = self.fff_name[result_index]
        # 'Banana', 'Oranges'
        if result_f == "Fresh" and result == "Apple":
            info_info = '\tCalories	Water\tProtein\tCarbs\tSugar\tFiber\tFat\nApple	52\t86%\t0.3g\t13.8g\t10.4g\t2.4g\t0.2g\nBenefit of Apple- Good for weight loss\n- Good for the heart\n- Lower risk of diabetes\n- Have prebiotic effects and promote good gut bacteria\n- Good for bone health\n- Protect against stomach injury\n- Can protect your brain\nEat 1-2 apples a day may lower cholesterol\n'
            self.label_info.setText(info_info)
        elif result_f == "Fresh" and result == "Banana":
            info_info = '\tCalories\tWater\tProtein\tCarbs\tSugar\tFiber\tFat\nBanana\t89\t75%\t1.1g\t22.8g\t12.2g\t2.6g\t0.3g\nBenefit of Banana\nFor Heart health, digestive health\n- Moderate blood sugar levels\n- Improve digestive health\n- Aid weight loss\n- Support heart health\n- Contain Powerful Antioxidants\n- Help you feel more full\nEat 1-2 bananas a day\n'
            self.label_info.setText(info_info)
        elif result_f == "Fresh" and result == "Oranges":
            info_info = '\tCalories\tWater\tProtein\tCarbs\tSugar\tFiber\tFat\nOrange\t47\t87%\t0.9g\t11.8g\t9.4g\t2.4g\t0.1g\nBenefit of Orange\nFor Heart health, Kidney stone prevention, anemia prevention, \nEat 1 orange a day\n'
            self.label_info.setText(info_info)
        else:
            self.label_info.setText("Don't eat!")
        self.result.setText(result)
        self.result_f.setText(result_f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())

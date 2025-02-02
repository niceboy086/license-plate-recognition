from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
#from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QEvent
from UI.ui_LicensePlate import Ui_MainWindow
import pyqtgraph
import numpy as np
import onnxruntime
import cv2
import os
import sys
# sys.path.append("../license_plate")
import onnx_infer
from PIL import Image, ImageDraw, ImageFont
import time


# 设置 PyQtGraph 显示配置
########################################################################################################################
# 设置显示背景色为白色，默认为黑色
pyqtgraph.setConfigOption('background', 'w')
# 设置显示前景色为黑色，默认为灰色
pyqtgraph.setConfigOption('foreground', 'k')
# 设置图像显示以行为主，默认以列为主
pyqtgraph.setConfigOption('imageAxisOrder', 'row-major')


class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.widget.ui.histogram.hide()
        self.widget.ui.menuBtn.hide()
        self.widget.ui.roiBtn.hide()
        self.currowcnt = 0
        self.rbtnstate = 0
        self.thrdtest = WorkerThread(self)
        self.vwthread = None
        self.label_11.setText('<a href="https://github.com/niceboy086/license-plate-recognition">https://github.com/niceboy086/license-plate-recognition</a>')
        self.label_13.setText('<a href="https://blog.csdn.net/ggw007">https://blog.csdn.net/ggw007</a>')
        self.label_11.setOpenExternalLinks(True)
        self.label_13.setOpenExternalLinks(True)
        self.label_11.installEventFilter(self)
        self.label_13.installEventFilter(self)
        self.labelcolor = self.label_11.palette().color(QPalette.Window)
        self.label_14.setText('47391900')
        self.label_14.setStyleSheet("background-color: yellow;color: blue;font: bold;" )
        #self.label_14.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label_14.installEventFilter(self)
        self.label_14.setToolTip('单击QQ号复制到剪贴板!')

        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setStyleSheet(
            "border-bottom-width: 0.5px;border-style: outset;border-color: rgb(220,220,220);")
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(
            ['序号', '图片文件', '车牌号','颜色','位置', '关键点', '置信度'])

        self.model_det_path = "model/plate_detect.onnx"
        self.model_rec_path = "model/plate_rec_color.onnx"
        providers = ['CPUExecutionProvider']
        self.session_detect = onnxruntime.InferenceSession(
                        self.model_det_path, providers=providers)
        self.session_rec = onnxruntime.InferenceSession(
                        self.model_rec_path, providers=providers)

        self.pushButton_2.clicked.connect(self.recFrom) # type: ignore
        # self.pushButton_3.clicked.connect(self.recFromFolder) # type: ignore
        #toggled信号与槽函数绑定
        self.radioButton_3.toggled.connect(lambda: self.btnstate(self.radioButton_3))
        self.radioButton_2.toggled.connect(lambda: self.btnstate(self.radioButton_2))
        self.radioButton.toggled.connect(lambda: self.btnstate(self.radioButton))
        self.thrdtest.signUi.connect(self.acceptthreadsignal)
        self.thrdtest.finished.connect(self.threadfinished)

    def eventFilter(self, obj, event):
        if obj == self.label_11:
            if event.type() == QEvent.Enter:
                print("label_11 mouse enter")
                self.label_11.setStyleSheet("background-color: yellow; ")
                return True  # 说明这个事件已被处理，其他控件别插手
            elif event.type() == QEvent.Leave:
                print("label_11 mouse leave")
                font = self.label_11.font()
                font.setBold(False)
                self.label_11.setFont(font)
                self.label_11.setStyleSheet("background-color: {self.labelcolor}; ")
                return True  # 说明这个事件已被处理，其他控件别插手
            elif event.type() == QEvent.MouseButtonPress:
                print("label_11 mouse MouseButtonPress")
                font = self.label_11.font()
                font.setBold(True)
                self.label_11.setFont(font)
                return True  # 说明这个事件已被处理，其他控件别插手
            elif event.type() == QEvent.MouseButtonRelease:
                print("label_11 mouse MouseButtonRelease")
                font = self.label_11.font()
                font.setBold(False)
                self.label_11.setFont(font)
                #return True  # 说明这个事件已被处理，其他控件别插手
        elif obj == self.label_13:
            if event.type() == QEvent.Enter:
                print("label_13 mouse enter")
                self.label_13.setStyleSheet("background-color: yellow; ")
                return True  # 说明这个事件已被处理，其他控件别插手
            elif event.type() == QEvent.Leave:
                print("label_13 mouse leave")
                font = self.label_13.font()
                font.setBold(False)
                self.label_13.setFont(font)
                self.label_13.setStyleSheet("background-color: {self.labelcolor}; ")
                return True  # 说明这个事件已被处理，其他控件别插手
            elif event.type() == QEvent.MouseButtonPress:
                print("label_13 mouse MouseButtonPress")
                font = self.label_13.font()
                font.setBold(True)
                self.label_13.setFont(font)
                return True  # 说明这个事件已被处理，其他控件别插手
            elif event.type() == QEvent.MouseButtonRelease:
                print("label_13 mouse MouseButtonRelease")
                font = self.label_13.font()
                font.setBold(False)
                self.label_13.setFont(font)
                #return True  # 说明这个事件已被处理，其他控件别插手
        elif obj == self.label_14:
            if event.type() == QEvent.MouseButtonRelease:
                print("label_14 mouse MouseButtonRelease")
                clipboard = QApplication.clipboard()
                clipboard.setText(self.label_14.text())
                #return True  # 说明这个事件已被处理，其他控件别插手

        return QMainWindow.eventFilter(self, obj, event)  # 交由其他控件处理

    def recFromPic(self, filename):
        #print("*debug, xxx*, RecFromPic:", filename)
        filename =filename.replace('\\', '/')
        pathseg = filename.split('/')[0:-2]
        #print(pathseg, type(pathseg))
        pathseg.append('result_onnx')
        #print(pathseg, type(pathseg))
        save_path = '/'.join(pathseg)
        #print(save_path)
        result_list, img0 = onnx_infer.det_rec_plate(
                    self.session_detect, self.session_rec, filename)
        #print("*debug, xxx*, result_list:", type(result_list))
        ori_img = onnx_infer.draw_result(img0, result_list)

        img_name = os.path.basename(filename)
        save_img_path = os.path.join(save_path,img_name)
        #print(save_img_path)
        cv2.imwrite(save_img_path,ori_img)

        return result_list, ori_img, filename

    def recFromMemPic(self, img):
        result_list, img0 = onnx_infer.det_rec_plate(
                    self.session_detect, self.session_rec, img)
        #print("*debug, xxx*, result_list:", type(result_list))
        ori_img = onnx_infer.draw_result(img0, result_list)

        return result_list, ori_img

    def platedisplay(self, filename, result_list, ori_img):
        self.widget.setImage(np.array(ori_img[:,:,::-1]))

        rowcnt = self.tableWidget.rowCount()
        #print(rowcnt)
        setboard = True

        for result in result_list:
            if self.currowcnt >= 5:
                self.tableWidget.insertRow(self.currowcnt)
                self.tableWidget.scrollToBottom()

            newItem = QTableWidgetItem(str(self.currowcnt+1))
            newItem.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(self.currowcnt, 0, newItem)
            if isinstance(filename, str):
                self.tableWidget.setItem(self.currowcnt, 1,
                        QTableWidgetItem(filename))
            else:
                self.tableWidget.setItem(self.currowcnt, 1,
                        QTableWidgetItem(f"frame:{filename}"))

            strrc = ','.join([str(int(result['rect'][0])),
                              str(int(result['rect'][1])),
                              str(int(result['rect'][2] - result['rect'][0])),
                              str(int(result['rect'][3] - result['rect'][1]))
                             ])

            newItem = QTableWidgetItem(result['plate_no'])
            newItem.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(self.currowcnt, 2,newItem)
            newItem = QTableWidgetItem(result['plate_color'])
            newItem.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(self.currowcnt, 3,newItem)
            self.tableWidget.setItem(self.currowcnt, 4,
                        QTableWidgetItem(strrc))

            #print(result['landmarks'])
            landmarks = [str(int(x)) for xy in result['landmarks'] for x in xy]
            landmarks = ','.join(landmarks)
            self.tableWidget.setItem(self.currowcnt, 5,
                        QTableWidgetItem(landmarks))
            newItem = QTableWidgetItem("{:.3f}".format(result['score']))
            newItem.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(self.currowcnt, 6,newItem )

            self.currowcnt += 1

            if setboard:
                self.lineEdit_2.setText(result['plate_no'])
                self.lineEdit_3.setText("{:.3f}".format(result['score']))
                self.lineEdit_4.setText(str(int(result['rect'][0])))
                self.lineEdit_5.setText(str(int(result['rect'][1])))
                self.lineEdit_6.setText(str(int(result['rect'][2])))
                self.lineEdit_7.setText(str(int(result['rect'][3])))
                setboard = False
    def recFrom(self):
        self.pushButton_2.setEnabled(False)
        print("recFrom::self.rbtnstate:", self.rbtnstate)
        if self.rbtnstate == 1:
            #print("*debug, xxx*, RecFromPic:", self.filename)
            pathseg = self.filename.split('/')[0:-2]
            #print(pathseg, type(pathseg))
            pathseg.append('result_onnx')
            #print(pathseg, type(pathseg))
            save_path = '/'.join(pathseg)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            result_list, ori_img, filename = self.recFromPic(self.filename)
            self.platedisplay(filename, result_list, ori_img)

            self.pushButton_2.setEnabled(True)

        elif self.rbtnstate == 2:
            begin = time.time()
            file_list = []
            onnx_infer.allFilePath(self.picdir, file_list)
            #print(file_list)
            pathseg = self.picdir.split('/')[0:-1]
            #print(pathseg, type(pathseg))
            pathseg.append('result_onnx')
            #print(pathseg, type(pathseg))
            save_path = '/'.join(pathseg)
            #print(save_path, type(save_path))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.file_list = file_list
            self.save_path = save_path
            print("thread started")
            self.thrdtest.start()
            #for pic_ in file_list:
            #    self.recFromPic(pic_)
            print(f"总共耗时{time.time()-begin} s")
        elif self.rbtnstate == 3:
            #print("*debug, xxx*, RecFromPic:", self.filename)
            pathseg = self.vfilename.split('/')[0:-2]
            #print(pathseg, type(pathseg))
            pathseg.append('result_videos')
            #print(pathseg, type(pathseg))
            save_path = '/'.join(pathseg)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.vsave_path = save_path
            if not self.vwthread:
                self.vwthread = VWorkerThread(self)
                self.vwthread.signUi.connect(self.acceptthreadsignal)
                self.vwthread.finished.connect(self.threadfinished)
            self.vwthread.start()

    def acceptthreadsignal(self, siginf):
        result_list, ori_img, filename = siginf
        self.platedisplay(filename, result_list, ori_img)
        print("acceptthreadsignal::acceptthreadsignal:", type(siginf))
    def threadfinished(self):
        self.thrdtest.wait()
        self.pushButton_2.setEnabled(True)
        print("threadfinised")

    @pyqtSlot()
    def on_pushButton_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "选择文件",
                        "D:/WORK/LicensePlate/imgs",
                        "Image Files (*.jpg *.gif *.png);;Text Files (*.txt);;All Files (*)",
                        options=options)

        if fileName:
            print(f"选择的文件：{fileName}")
            self.lineEdit.setText(fileName)
            image = Image.open(fileName)
            if image is not None:
                # 如果之前未设置显示选项以行为主，这里需要对显示图像进行转置
                self.widget.setImage(np.array(image))

                pix = QPixmap(fileName)
                self.label_3.setPixmap(pix)
                self.label_3.setScaledContents(True)  # 自适应QLabel大小
                self.filename = fileName

                self.radioButton_2.setChecked(True)
        else:
            self.lineEdit.setPlaceholderText("none")

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        dir = QFileDialog.getExistingDirectory(self,
                        "选取文件夹", "D:/WORK/LicensePlate/imgs")  # 起始路径
        if dir:
            print(f"选择的文件夹：{dir}")
            self.lineEdit_8.setText(dir)
            self.radioButton.setChecked(True)
            self.picdir = dir

    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "选择视频文件",
                        "D:/WORK/LicensePlate/videos",
                        "video Files (*.mp4);;All Files (*)",
                        options=options)

        if fileName:
            print(f"选择的视频文件：{fileName}")
            self.lineEdit_9.setText(fileName)
            capture = cv2.VideoCapture(fileName)
            if capture.isOpened():
                ret,img=capture.read()
                if ret:
                    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    # 如果之前未设置显示选项以行为主，这里需要对显示图像进行转置
                    self.widget.setImage(np.array(image))

                self.vfilename = fileName
                self.radioButton_3.setChecked(True)
            capture.release()
        else:
            self.lineEdit.setPlaceholderText("none")


    def btnstate(self, btn):
        # 输出按钮1与按钮2的状态，选中还是没选中
        if btn.isChecked() == True:
            if btn.objectName() == "radioButton_2":
                self.rbtnstate = 1
            elif btn.objectName() == "radioButton":
                self.rbtnstate = 2
            elif btn.objectName() == "radioButton_3":
                self.rbtnstate = 3

        print("rbtnstate:", self.rbtnstate)

class WorkerThread(QThread):
    signUi = pyqtSignal(list)
    def __init__(self, param, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.param = param
        self.working = True
        self.num = 0

    def __del__(self):
        self.working = False
        self.wait()

    def run(self):
        for pic_ in self.param.file_list:
            result_list, ori_img, filename = self.param.recFromPic(pic_)
            self.signUi.emit([result_list, ori_img, filename])
            #self.sleep(3)
        # print(type(self.param))

class VWorkerThread(QThread):
    signUi = pyqtSignal(list)
    def __init__(self, param, parent=None):
        super(VWorkerThread, self).__init__(parent)
        self.param = param
        self.working = True
        self.num = 0

    def __del__(self):
        self.working = False
        self.wait()

    def run(self):
        capture=cv2.VideoCapture(self.param.vfilename)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        save_file_path = os.path.join(self.param.vsave_path, 'result.mp4')
        print(save_file_path)
        out = cv2.VideoWriter(save_file_path, fourcc, fps, (width, height))  # 写入视频
        frame_count = 0
        fps_all=0
        rate,FrameNumber,duration=onnx_infer.get_second(capture)
        if capture.isOpened():
            while True:
                t1 = cv2.getTickCount()
                frame_count += 1
                print(f"第{frame_count} 帧", end=" ")
                ret, img = capture.read()
                if not ret:
                    break
                result_list, ori_img = self.param.recFromMemPic(img)
                # if frame_count%rate==0:
                t2 = cv2.getTickCount()
                infer_time = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / infer_time
                fps_all += fps
                str_fps = f'fps:{fps:.4f}'

                cv2.putText(ori_img, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.imshow("haha",ori_img)
                # cv2.waitKey(1)
                out.write(ori_img)
                self.signUi.emit([result_list, ori_img, frame_count])

        else:
            print("失败")
        capture.release()
        out.release()


if __name__ == '__main__':
    img_size = (640, 640)
    app = QApplication(sys.argv)
    vieo_gui = MainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())
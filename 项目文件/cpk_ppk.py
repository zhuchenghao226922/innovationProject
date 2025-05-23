# Form implementation generated from reading ui file 'cpk_ppk.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QMainWindow
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(435, 317)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(150, 190, 71, 16))
        self.label_4.setObjectName("label_4")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(160, 130, 181, 51))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 240, 75, 24))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(150, 210, 71, 16))
        self.label_5.setObjectName("label_5")
        self.radioButton_2 = QtWidgets.QRadioButton(parent=self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(150, 110, 101, 20))
        self.radioButton_2.setObjectName("radioButton_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(230, 60, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 220, 75, 24))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(350, 270, 75, 24))
        self.pushButton_3.setObjectName("pushButton_3")
        self.listView = QtWidgets.QListView(parent=self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(10, 10, 111, 201))
        self.listView.setProperty("isWrapping", False)
        self.listView.setObjectName("listView")
        self.radioButton = QtWidgets.QRadioButton(parent=self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(150, 30, 71, 20))
        self.radioButton.setObjectName("radioButton")
        self.lineEdit = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(230, 30, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(240, 210, 113, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.verticalScrollBar = QtWidgets.QScrollBar(parent=self.centralwidget)
        self.verticalScrollBar.setGeometry(QtCore.QRect(320, 130, 16, 51))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(150, 80, 101, 16))
        self.label_3.setObjectName("label_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(240, 190, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(150, 60, 71, 16))
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(150, 10, 61, 16))
        self.label.setScaledContents(False)
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.radioButton.clicked.connect(self.on_radioButton_clicked)
        self.radioButton_2.clicked.connect(self.on_radioButton_2_clicked)
        self.pushButton.clicked.connect(self.on_pushButton_clicked)  # 新增：连接 pushButton 的点击事件

        # 设置 radioButton 为默认点击状态
        self.radioButton.setChecked(True)
        self.on_radioButton_clicked()

        # 新增：连接 pushButton_3 的点击事件，用于关闭窗口
        self.pushButton_3.clicked.connect(self.on_pushButton_3_clicked)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "规格下限(L):"))
        self.pushButton_2.setText(_translate("MainWindow", "确定"))
        self.label_5.setText(_translate("MainWindow", "规格上限(U):"))
        self.radioButton_2.setText(_translate("MainWindow", "子组跨数列(B):"))
        self.pushButton.setText(_translate("MainWindow", "选择"))
        self.pushButton_3.setText(_translate("MainWindow", "取消"))
        self.radioButton.setText(_translate("MainWindow", "单列(C):"))
        self.label_3.setText(_translate("MainWindow", "(使用常量或ID列)"))
        self.label_2.setText(_translate("MainWindow", "子组大小(Z):"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p>数据排列为</p></body></html>"))

    def __init__(self, main_window):
        super().__init__()
        # 保存主窗口对象
        self.main_window = main_window
        self.all_table_data = []  # 用于存储所有表格数据
        self.setupUi(self)
        self.get_table_data()
     #获取主窗口调用的数据-------------------------------------

    def get_table_data(self):
        table = self.main_window.tableWidget
        rows = table.rowCount()
        columns = table.columnCount()
        print(f"表格行数: {rows}, 列数: {columns}")  # 添加调试信息

        data = []
        for row in range(rows):
            row_data = []
            for col in range(columns):
                item = table.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append(None)
            data.append(row_data)
        print("获取到的主窗口表格数据:", data)

        # 筛选出有数据的列
        valid_columns = []
        for col in range(columns):
            has_data = False
            for row in range(rows):
                if data[row][col] is not None and data[row][col].strip():
                    has_data = True
                    break
            if has_data:
                valid_columns.append(col)

        # 生成列名
        column_names = [f"c{i + 1}" for i in valid_columns]

        # 创建模型并设置数据
        model = QtGui.QStandardItemModel()
        for name in column_names:
            item = QtGui.QStandardItem(name)
            model.appendRow(item)

        # 将模型设置给 listView
        self.listView.setModel(model)
        self.all_table_data = data  # 保存所有表格数据

    # 单列按钮点击事件
    def on_radioButton_clicked(self):
        self.lineEdit.setEnabled(True)
        self.lineEdit_2.setEnabled(True)
        self.plainTextEdit.setEnabled(False)

    # 子组跨数列按钮点击事件
    def on_radioButton_2_clicked(self):
        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.plainTextEdit.setEnabled(True)

    # 新增：pushButton 点击事件处理函数
    def on_pushButton_clicked(self):
        if self.radioButton.isChecked():  # 确保 radioButton 为点击状态
            selected_indexes = self.listView.selectedIndexes()
            if selected_indexes:
                selected_column_names = [index.data() for index in selected_indexes]
                # 将选中的列名显示在 lineEdit 文本框中
                self.lineEdit.setText(", ".join(selected_column_names))
        elif self.radioButton_2.isChecked():  # 确保 radioButton_2 为点击状态
            selected_indexes = self.listView.selectedIndexes()
            if selected_indexes:
                selected_column_names = [index.data() for index in selected_indexes]
                # 获取当前 plainTextEdit 中的内容
                current_text = self.plainTextEdit.toPlainText()
                # 将当前内容按逗号分割，去除重复的列名
                existing_names = [name.strip() for name in current_text.split(',') if name.strip()]
                # 将新选中的列名与已有的列名合并，并去重
                combined_names = list(set(existing_names + selected_column_names))
                # 将去重后的列名按逗号分隔，设置到 plainTextEdit 中
                self.plainTextEdit.setPlainText(", ".join(combined_names))

    # 新增：pushButton_2 点击事件处理函数
    def on_pushButton_2_clicked(self):
        if self.radioButton.isChecked():
            # 获取 lineEdit 中的列名
            column_names = self.lineEdit.text().strip().split(',')
            column_names = [name.strip() for name in column_names if name.strip()]

            # 从 all_table_data 中提取相应的列数据
            dataset = []
            for col_name in column_names:
                col_index = int(col_name[1:]) - 1  # 假设列名格式为 c1, c2, ...
                column_data = [float(row[col_index]) for row in self.all_table_data if
                               row[col_index] is not None and row[col_index].strip()]
                dataset.extend(column_data)  # 将列数据添加到数据集中

            # 检查数据集是否为空
            if not dataset:
                print("错误：没有有效的数据可供计算。")
                return

            # 将数据集转换为 NumPy 数组，并确保数据类型为 float
            try:
                dataset = np.array(dataset, dtype=float)
            except ValueError as e:
                print(f"数据转换错误: {e}")
                return

            # 获取 lineEdit_2 中的子组大小
            try:
                n = int(self.lineEdit_2.text())
            except ValueError as e:
                print(f"子组大小转换错误: {e}")
                return

            x = split_with_padding(dataset, n)
            xi, r = xi_r(x)
            # n为x的维数
            n = x.shape[1]
            #计算平均值
            mean = np.mean(x)
            # 计算子组中的极差的平均值
            r_values = np.mean(r)
            # 使用样本标准差
            std_dev = np.std(x, ddof=1)
            # 计算过程标准差
            sigma = calculate_sigma_r(r_values, n)
            # 将 lsl 和 usl 转换为 float 类型
            try:
                lsl = float(self.lineEdit_3.text())
                usl = float(self.lineEdit_4.text())
            except ValueError as e:
                print(f"规格限转换错误: {e}")
                return
            cp = calculate_cp(usl, lsl, sigma)
            cpk = calculate_cpk(usl, lsl, mean, sigma)
            pp = calculate_pp(usl, lsl, std_dev)
            ppk = calculate_ppk(usl, lsl, mean, std_dev)
            print(cp, cpk, pp, ppk)
            hist(dataset, std_dev,lsl,usl,cp,cpk,pp,ppk)
            self.close()

        elif self.radioButton_2.isChecked():
            # 获取 plainTextEdit 中的列名
            column_names = self.plainTextEdit.toPlainText().strip().split(',')
            column_names = [name.strip() for name in column_names if name.strip()]
            
            # 从 all_table_data 中提取相应的列数据
            n_dimensional_data = []
            for col_name in column_names:
                col_index = int(col_name[1:]) - 1  # 假设列名格式为 c1, c2, ...
                column_data = [float(row[col_index]) for row in self.all_table_data if
                               row[col_index] is not None and row[col_index].strip()]
                n_dimensional_data.append(column_data)

            # 对 n_dimensional_data 进行转置操作
            x = list(map(list, zip(*n_dimensional_data)))

            # 检查 x 是否为空
            if not x:
                print("错误：没有有效的数据可供计算。")
                return

            try:
                n_dimensional_data = np.array(n_dimensional_data, dtype=float)
            except ValueError as e:
                print(f"数据转换错误: {e}")
                return

            # 将 x 转换为 NumPy 数组，并确保数据类型为 float
            try:
                x = np.array(x, dtype=float)
            except ValueError as e:
                print(f"数据转换错误: {e}")
                return

            # 调用 xi_r 方法
            try:
                xi, r = xi_r(x)
            except Exception as e:
                print(f"计算过程中发生错误: {e}")
                return
            #将x转化为一维数组
            x_flattened = np.array(x).flatten()

            xi, r = xi_r(x)
            # n为x的维数
            n = x.shape[1]
            #计算平均值
            mean = np.mean(x)
            # 计算子组中的极差的平均值
            r_values = np.mean(r)
            # 使用样本标准差
            std_dev = np.std(x, ddof=1)
            # 计算过程标准差
            sigma = calculate_sigma_r(r_values, n)
            # 将 lsl 和 usl 转换为 float 类型
            try:
                lsl = float(self.lineEdit_3.text())
                usl = float(self.lineEdit_4.text())
            except ValueError as e:
                print(f"规格限转换错误: {e}")
                return
            cp = calculate_cp(usl, lsl, sigma)
            cpk = calculate_cpk(usl, lsl, mean, sigma)
            pp = calculate_pp(usl, lsl, std_dev)
            ppk = calculate_ppk(usl, lsl, mean, std_dev)
            print(cp, cpk, pp, ppk)
            hist(x_flattened, std_dev,lsl,usl,cp,cpk,pp,ppk)
            self.close()

    def on_pushButton_3_clicked(self):
        # 关闭子界面
        self.close()

def split_with_padding(aa, n):
    # 计算需要补充的元素个数
    pad = (n - (len(aa) % n)) % n
    # 计算平均值
    mean_val = aa.mean()
    # 生成补充元素数组
    padding = np.full(pad, mean_val)
    # 拼接原数组和补充元素
    padded_arr = np.concatenate([aa, padding])
    # 重塑为二维数组
    return padded_arr.reshape(-1, n)

def xi_r(x):
    # 计算每组平均值
    xi = [np.mean(subgroup) for subgroup in x]
    # 计算每组的极差r
    r = [np.ptp(subgroup) for subgroup in x]
    return xi, r

# 计算CP和CPK
def calculate_cp(usl, lsl, sigma):
    return abs((usl - lsl) / (6 * sigma))

def calculate_cpk(usl, lsl, mean, sigma):
    cpu = abs((usl - mean) / (3 * sigma))
    cpl = abs((mean - lsl) / (3 * sigma))
    return min(cpu, cpl)

# 计算PP和PPK
def calculate_pp(usl, lsl, std_dev):
    return abs((usl - lsl) / (6 * std_dev))

def calculate_ppk(usl, lsl, mean, std_dev):
    ppu = abs((usl - mean) / (3 * std_dev))
    ppl = abs((mean - lsl) / (3 * std_dev))
    return min(ppu, ppl)

# 计算过程标准差
def calculate_sigma_r(R_values, n=None):
    D2_TABLE = {
        2: 1.128, 3: 1.693, 4: 2.059,
        5: 2.326, 6: 2.534, 7: 2.704,
        8: 2.847, 9: 2.970, 10: 3.078
    }
    # 自动推断子组大小（如果未指定）
    inferred_n = n or infer_subgroup_size(R_values)

    # 获取d2值
    try:
        d2 = D2_TABLE[inferred_n]
    except KeyError:
        raise ValueError(f"未找到n={inferred_n}对应的d2值，支持的子组大小：{list(D2_TABLE.keys())}")

    # 计算R_bar
    R_bar = np.mean(R_values)

    # 计算过程标准差
    return R_bar / d2

# 绘制直方图和正态分布曲线
def hist(x,std_dev,LSL,USL,CP,CPK,PP,PPK):
    num_bins = 12
    u = x.mean()
    n, bins, patches = plt.hist(x, num_bins, density=1, facecolor='blue', alpha=0.5)
    plt.axvline(x=USL, color='r', linestyle='--', label='USL')
    plt.axvline(x=LSL, color='r', linestyle='--', label='LSL')
    y = stats.norm.pdf(bins, u, std_dev)
    plt.plot(bins, y, 'r--')
    plt.subplots_adjust(left=0.05)

    plt.text(0.15, 0.95, 'CP: {:.2f}'.format(CP), fontsize=15, transform=plt.gca().transAxes, va='top',ha='center')
    plt.text(0.15, 0.9, 'CPK: {:.2f}'.format(CPK), fontsize=15, transform=plt.gca().transAxes, va='top',ha='center')
    plt.text(0.15, 0.85, 'PP: {:.2f}'.format(PP),  fontsize=15, transform=plt.gca().transAxes, va='top',ha='center')
    plt.text(0.15, 0.8, 'PPK: {:.2f}'.format(PPK),  fontsize=15, transform=plt.gca().transAxes, va='top',ha='center')
    plt.text(0.15, 0.75, 'LSL: {:.2f}'.format(LSL), fontsize=15, transform=plt.gca().transAxes, va='top',ha='center')
    plt.text(0.15, 0.7, 'USL: {:.2f}'.format(USL),  fontsize=15, transform=plt.gca().transAxes, va='top',ha='center')

    plt.show()

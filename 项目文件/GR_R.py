

from PyQt6 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListView, QRadioButton, QLineEdit, QComboBox,
    QCheckBox, QPushButton, QLabel, QFileDialog, QMessageBox, QMenuBar, QStatusBar, QTableWidgetItem
)
class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(570, 301)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listView = QtWidgets.QListView(parent=self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(20, 10, 191, 192))
        self.listView.setObjectName("listView")
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 220, 75, 24))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_5 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(450, 40, 75, 24))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(450, 70, 75, 24))
        self.pushButton_6.setObjectName("pushButton_6")
        self.radioButton = QtWidgets.QRadioButton(parent=self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(230, 200, 93, 20))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(parent=self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(230, 220, 93, 20))
        self.radioButton_2.setObjectName("radioButton_2")
        self.pushButton_7 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(450, 240, 75, 24))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(450, 210, 75, 24))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(20, 250, 75, 24))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(230, 180, 54, 16))
        self.label_4.setObjectName("label_4")
        self.widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(230, 30, 199, 74))
        self.widget.setObjectName("widget")
        self.formLayout = QtWidgets.QFormLayout(self.widget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(parent=self.widget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label)
        self.lineEdit = QtWidgets.QLineEdit(parent=self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lineEdit)
        self.label_2 = QtWidgets.QLabel(parent=self.widget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lineEdit_2)
        self.label_3 = QtWidgets.QLabel(parent=self.widget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_3)
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.widget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lineEdit_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 连接按钮点击事件
        self.pushButton.clicked.connect(self.on_select_button_clicked)
        self.pushButton_7.clicked.connect(self.on_cancel_button_clicked)
        self.pushButton_8.clicked.connect(self.on_confirm_button_clicked)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "选择"))
        self.pushButton_5.setText(_translate("MainWindow", "量具信息"))
        self.pushButton_6.setText(_translate("MainWindow", "选项"))
        self.radioButton.setText(_translate("MainWindow", "方差分析"))
        self.radioButton_2.setText(_translate("MainWindow", "Xbar 和 R"))
        self.pushButton_7.setText(_translate("MainWindow", "取消"))
        self.pushButton_8.setText(_translate("MainWindow", "确定"))
        self.pushButton_9.setText(_translate("MainWindow", "帮助"))
        self.label_4.setText(_translate("MainWindow", "分析方法"))
        self.label.setText(_translate("MainWindow", "部件号："))
        self.label_2.setText(_translate("MainWindow", "操作员："))
        self.label_3.setText(_translate("MainWindow", "测试数据："))

    def __init__(self, main_window):
        super().__init__()
        # 保存主窗口对象
        self.main_window = main_window
        self.all_table_data = []  # 用于存储所有表格数据
        self.setupUi(self)
        self.get_table_data()

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

    def on_select_button_clicked(self):
        try:
            current_index = self.listView.currentIndex()
            if current_index.isValid():
                model = self.listView.model()
                selected_item = model.itemFromIndex(current_index)
                if selected_item is not None:
                    selected_text = selected_item.text()
                    if not self.lineEdit.text():
                        self.lineEdit.setText(selected_text)
                    elif not self.lineEdit_2.text():
                        self.lineEdit_2.setText(selected_text)
                    elif not self.lineEdit_3.text():
                        self.lineEdit_3.setText(selected_text)
                else:
                    print("No item selected in listView.")
            else:
                print("No valid index selected in listView.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def on_confirm_button_clicked(self):
        # 获取 lineEdit 中的列名
        column_napimes = self.lineEdit.text().strip().split(',')
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

        # 获取 lineEdit_2 中的列名
        column_names_2 = self.lineEdit_2.text().strip().split(',')
        column_names_2 = [name.strip() for name in column_names_2 if name.strip()]

        # 从 all_table_data 中提取相应的列数据
        dataset_2 = []
        for col_name_2 in column_names_2:
            col_index_2 = int(col_name_2[1:]) - 1  # 假设列名格式为 c1, c2, ...
            column_data_2 = [float(row[col_index_2]) for row in self.all_table_data if
                           row[col_index_2] is not None and row[col_index_2].strip()]
            dataset_2.extend(column_data_2)  # 将列数据添加到数据集中

        # 检查数据集是否为空
        if not dataset_2:
            print("错误：没有有效的数据可供计算。")
            return

        # 将数据集转换为 NumPy 数组，并确保数据类型为 float
        try:
            dataset_2 = np.array(dataset_2, dtype=float)
        except ValueError as e:
            print(f"数据转换错误: {e}")
            return

        # 获取 lineEdit_3 中的列名
        column_names_3 = self.lineEdit_3.text().strip().split(',')
        column_names_3 = [name.strip() for name in column_names_3 if name.strip()]

        # 从 all_table_data 中提取相应的列数据
        dataset_3 = []
        for col_name_3 in column_names_3:
            col_index_3 = int(col_name_3[1:]) - 1  # 假设列名格式为 c1, c2, ...
            column_data_3 = [float(row[col_index_3]) for row in self.all_table_data if
                           row[col_index_3] is not None and row[col_index_3].strip()]
            dataset_3.extend(column_data_3)  # 将列数据添加到数据集中

        # 检查数据集是否为空
        if not dataset_3:
            print("错误：没有有效的数据可供计算。")
            return

        # 将数据集转换为 NumPy 数组，并确保数据类型为 float
        try:
            dataset_3 = np.array(dataset_3, dtype=float)
        except ValueError as e:
            print(f"数据转换错误: {e}")
            return

        data = data_list_to_dict(dataset, dataset_2, dataset_3)
        # 创建 DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"数据转换错误: {e}")
            return

        anova_table,results = GR_R(data)
        print(anova_table)
        print(results)
        try:
            anova_table_str = anova_table.to_string()
            self.main_window.display_text(anova_table_str+"\n")
        except Exception as e:
            print(f"数据转换错误: {e}")
            return

        try:
            results_str = results.to_string()
            self.main_window.display_text(results_str+"\n")
        except Exception as e:
            print(f"数据转换错误: {e}")
            return

        self.close()



    def on_cancel_button_clicked(self):
        self.close()



def GR_R(data):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 进行方差分析 ANOVA
    model = ols('Measurement ~ C(Operator) + C(Part) + C(Operator):C(Part)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)

    # 计算各部分方差
    MS_operator = anova_table['sum_sq']['C(Operator)'] / anova_table['df']['C(Operator)']
    MS_parts = anova_table['sum_sq']['C(Part)'] / anova_table['df']['C(Part)']
    MS_interaction = anova_table['sum_sq']['C(Operator):C(Part)'] / anova_table['df']['C(Operator):C(Part)']
    MS_repeatability = anova_table['sum_sq']['Residual'] / anova_table['df']['Residual']

    #计算零件个数
    parts_count = len(df['Part'].unique())

    #计算人数
    operator_count = len(df['Operator'].unique())

    #计算测量总次数
    total_measurements = len(df)

    #计算观测次数
    r = total_measurements / parts_count / operator_count

    # 计算标准差
    #EV
    EV_repeatability = MS_repeatability
    SD_repeatability = np.sqrt(MS_repeatability)

    EV_operator = (MS_operator-MS_interaction)/(r*parts_count)
    SD_operator = np.sqrt((MS_operator-MS_interaction)/(r*parts_count))
    #PV
    EV_parts = (MS_parts - MS_interaction) / (r * operator_count)
    SD_parts = np.sqrt((MS_parts - MS_interaction) / (r * operator_count))

    EV_interaction = (MS_interaction - MS_repeatability) / r
    SD_interaction = np.sqrt((MS_interaction - MS_repeatability) / r)
    #AV
    EV_reproducibility = EV_operator + EV_interaction
    SD_reproducibility = np.sqrt(EV_reproducibility)
    #R&R
    R_and_R = np.sqrt(SD_repeatability ** 2 + SD_reproducibility ** 2)
    #TV
    GR_and_R = np.sqrt(SD_repeatability ** 2 + SD_reproducibility ** 2 + SD_parts ** 2)

    # 计算研究变异
    R_and_R_percent = (R_and_R / GR_and_R) * 100
    SV_repeatability = (SD_repeatability / GR_and_R) * 100
    SV_reproducibility = (SD_reproducibility / GR_and_R) * 100
    SV_operator = (SD_operator / GR_and_R) * 100
    SV_interaction = (SD_interaction / GR_and_R) * 100
    SV_parts = (SD_parts / GR_and_R) * 100

    # 构建结果字典
    results = {
        "来源": ["合计量具 R&R", "重复性", "再现性", "operators", "operators*parts", "部件间", "合计变异"],
        "标准差(SD)": [R_and_R, SD_repeatability, SD_reproducibility, SD_operator, SD_interaction, SD_parts, GR_and_R],
        "研究变异(6 * SD)": [6 * R_and_R, 6 * SD_repeatability, 6 * SD_reproducibility, 6 * SD_operator, 6 * SD_interaction, 6 * SD_parts, 6 * GR_and_R],
        "%研究变异(%SV)": [R_and_R_percent, SV_repeatability, SV_reproducibility, SV_operator, SV_interaction, SV_parts, 100.00]
    }
    result_df = pd.DataFrame(results)
    return anova_table,result_df

def P_T_test(data):
    result = GR_R(data)

    #提出R_and_R
    R_and_R = result.loc[result['来源'] == '合计量具 R&R', '标准差(SD)'].values[0]

    #提出GR_and_R
    GR_and_R = result.loc[result['来源'] == '合计变异', '标准差(SD)'].values[0]

    #计算P/TV测量系统能力
    P_TV = R_and_R / GR_and_R

    #让用户输入上公差USL与下公差LSL
    USL = float(input("请输入上公差USL："))
    LSL = float(input("请输入下公差LSL："))

    #计算P/T测量系统能力
    P_T = R_and_R / (USL - LSL)
    print("P/TV测量系统能力为：", P_TV)
    print("P/T测量系统能力为：", P_T)

    #计算测量系统的分辨力


    #判断测量系统能力
    if P_TV < 0.1 or P_T < 0.1:
        print("测量系统能力很好")
    elif 0.1 <= P_TV <= 0.3 or 0.1 <= P_T <= 0.3:
        print("测量系统能力处于临界状态")
    elif P_TV > 0.3 or P_T > 0.3:
        print("测量系统能力不足，必须加以改进")

def data_list_to_dict(part_list, operator_list, measurement_list):
    data = {
        'Part': part_list,
        'Operator': operator_list,
        'Measurement': measurement_list
    }
    return data
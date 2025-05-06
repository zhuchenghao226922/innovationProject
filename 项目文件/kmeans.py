import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListView, QRadioButton, QLineEdit, QComboBox,
    QCheckBox, QPushButton, QLabel, QFileDialog, QMessageBox, QMenuBar, QStatusBar, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QStringListModel
from PyQt6 import QtCore, QtGui
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(413, 423)

        self.centralwidget = QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 左侧列表视图
        self.listView_11 = QListView(parent=self.centralwidget)
        self.listView_11.setGeometry(QtCore.QRect(10, 10, 121, 271))
        self.listView_11.setObjectName("listView_11")
        self.listView_11.setSelectionMode(QListView.SelectionMode.MultiSelection)  # 允许多选

        # 右侧列表视图
        self.listView_12 = QListView(parent=self.centralwidget)
        self.listView_12.setGeometry(QtCore.QRect(140, 10, 260, 40))
        self.listView_12.setObjectName("listView_12")
        self.listView_12.setModel(QStringListModel())  # 初始化模型

        # 删除右侧列表项的按钮
        self.delete_button = QPushButton("删除", parent=self.centralwidget)
        self.delete_button.setGeometry(QtCore.QRect(230, 50, 80, 20))
        self.delete_button.clicked.connect(self.delete_selected_columns)

        # 聚类数单选按钮
        self.cluster_var = QRadioButton("指定聚类数", parent=self.centralwidget)
        self.cluster_var.setGeometry(QtCore.QRect(140, 70, 100, 20))
        self.cluster_var.setChecked(True)
        self.cluster_var.toggled.connect(self.toggle_input)

        # 聚类数输入框
        self.cluster_label = QLabel("聚类数:", parent=self.centralwidget)
        self.cluster_label.setGeometry(QtCore.QRect(140, 100, 50, 20))
        self.cluster_entry = QLineEdit("3", parent=self.centralwidget)
        self.cluster_entry.setGeometry(QtCore.QRect(200, 100, 80, 20))
        self.cluster_entry.setEnabled(True)

        # 初始列单选按钮
        self.column_var = QRadioButton("指定初始列", parent=self.centralwidget)
        self.column_var.setGeometry(QtCore.QRect(140, 130, 130, 20))
        self.column_var.toggled.connect(self.toggle_input)

        # 初始列下拉框
        self.column_label = QLabel("初始列:", parent=self.centralwidget)
        self.column_label.setGeometry(QtCore.QRect(140, 160, 50, 20))
        self.column_combobox = QComboBox(parent=self.centralwidget)
        self.column_combobox.setGeometry(QtCore.QRect(200, 160, 120, 20))
        self.column_combobox.setEnabled(False)

        # 标准化数据复选框
        self.std_checkbox = QCheckBox("标准化数据", parent=self.centralwidget)
        self.std_checkbox.setGeometry(QtCore.QRect(140, 190, 100, 20))
        self.std_checkbox.setChecked(True)

        # 数据加载按钮和标签
        self.load_button = QPushButton("加载文件", parent=self.centralwidget)
        self.load_button.setGeometry(QtCore.QRect(140, 220, 80, 20))
        self.load_button.clicked.connect(self.load_data)
        self.file_label = QLabel("未选择文件", parent=self.centralwidget)
        self.file_label.setGeometry(QtCore.QRect(230, 220, 150, 20))


        self.pushButton = QPushButton("选择", parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 300, 75, 23))
        self.pushButton.clicked.connect(self.select_columns)
        self.pushButton_5 = QPushButton("运行分析", parent=self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(230, 300, 81, 23))
        self.pushButton_5.clicked.connect(self.run_analysis)
        self.pushButton_6 = QPushButton("帮助", parent=self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 340, 75, 23))
        self.pushButton_4 = QPushButton("退出", parent=self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(320, 300, 81, 23))
        self.pushButton_4.clicked.connect(self.close)

        MainWindow.setCentralWidget(self.centralwidget)

        # 菜单栏和状态栏
        self.menubar = QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 413, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(parent=MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "K 均值聚类分析"))

    def toggle_input(self):
        """切换输入控件状态"""
        if self.cluster_var.isChecked():
            self.cluster_entry.setEnabled(True)
            self.column_combobox.setEnabled(False)
        else:
            self.cluster_entry.setEnabled(False)
            self.column_combobox.setEnabled(True)

    def load_data(self):
        """加载数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "Excel文件 (*.xls *.xlsx);;CSV文件 (*.csv);;所有文件 (*.*)")
        if not file_path:
            return

        try:
            if file_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path, engine='openpyxl')
            else:
                self.data = pd.read_csv(file_path)

            self.file_label.setText(file_path.split("/")[-1])
            self.column_combobox.clear()
            self.column_combobox.addItems(self.data.columns)

            # 将数据列显示在左侧列表视图
            model = QStringListModel(self.data.columns.tolist())
            self.listView_11.setModel(model)

            # 清空右侧列表视图
            self.selected_columns = []
            model = QStringListModel(self.selected_columns)
            self.listView_12.setModel(model)

            QMessageBox.information(self, "加载成功", f"成功加载{len(self.data)}行数据")

        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"文件加载失败: {str(e)}")

    def get_table_data(self):
        try:
            if not hasattr(self.main_window, 'tableWidget'):
                QMessageBox.critical(self, "错误", "无法访问主窗口表格")
                return

            table = self.main_window.tableWidget
            rows = table.rowCount()
            cols = table.columnCount()

            # 获取列名（第一行）
            column_names = []
            for col in range(cols):
                item = table.item(0, col)
                col_name = item.text().strip() if item and item.text().strip() else f"列{col+1}"
                column_names.append(col_name)

            # 处理数据行（从第二行开始）
            numeric_data = []
            valid_columns = []
            for col in range(cols):
                col_values = []
                has_numeric = False
                for row in range(1, rows):  # 跳过标题行
                    item = table.item(row, col)
                    if item and item.text().strip():
                        try:
                            val = float(item.text())
                            col_values.append(val)
                            has_numeric = True
                        except ValueError:
                            col_values.append(np.nan)
                    else:
                        col_values.append(np.nan)

                if has_numeric:  # 仅保留包含数值的列
                    numeric_data.append(col_values)
                    valid_columns.append(col)

            # 创建DataFrame
            self.data = pd.DataFrame(
                np.array(numeric_data).T,
                columns=[column_names[col] for col in valid_columns]
            )
            # 创建DataFrame后添加以下代码
            self.file_label.setText("从主界面导入")
            self.column_combobox.clear()
            self.column_combobox.addItems(self.data.columns.tolist())  # 新增这行

            # 更新列选择列表
            model = QStringListModel(self.data.columns.tolist())
            self.listView_11.setModel(model)
            self.selected_columns.clear()
            self.listView_12.model().setStringList([])
        except Exception as e:
            return
    def select_columns(self):
        """选择列并显示在右侧列表视图"""
        indexes = self.listView_11.selectedIndexes()
        for index in indexes:
            column = index.data()
            if column not in self.selected_columns:
                self.selected_columns.append(column)

        # 更新右侧列表视图
        model = QStringListModel(self.selected_columns)
        self.listView_12.setModel(model)

    def delete_selected_columns(self):
        """删除右侧列表视图中选中的列"""
        indexes = self.listView_12.selectedIndexes()
        for index in sorted(indexes, reverse=True):
            column = index.data()
            self.selected_columns.remove(column)

        # 更新右侧列表视图
        model = QStringListModel(self.selected_columns)
        self.listView_12.setModel(model)

    def run_analysis(self):
        if self.data is None or self.data.empty:
            QMessageBox.critical(self, "没有数据", "请先输入数据或加载文件")
            return

        if not self.selected_columns:
            QMessageBox.critical(self, "没有选择参与聚类分析的列", "请选择参与聚类分析的列")
            return

        try:
            # 数据预处理
            raw_data = self.data[self.selected_columns].select_dtypes(include=np.number)
            raw_data = raw_data.dropna()
            if len(raw_data) == 0:
                raise ValueError("参与聚类的数据均为缺失值或非数值型")
            X = raw_data.values

            # 标准化处理
            if self.std_checkbox.isChecked():
                original_means = np.mean(X, axis=0)  # 存储原始均值
                means = np.mean(X, axis=0)
                stds = np.std(X, axis=0, ddof=1)
                stds[stds == 0] = 1.0
                X_normalized = (X - means) / stds
            else:
                original_means = np.mean(X, axis=0)  # 始终计算原始均值
                X_normalized = X.copy()
                means = np.zeros(X.shape[1])
                stds = np.ones(X.shape[1])

            # 初始化参数
            if self.cluster_var.isChecked():
                k = int(self.cluster_entry.text())
                centroids_normalized, clusters_normalized = self.k_means(X_normalized, k)
            else:
                init_col = self.column_combobox.currentText()
                init_series = self.data.loc[raw_data.index, init_col]

                # 验证初始列
                if init_series.isnull().any():
                    raise ValueError(f"初始列 '{init_col}' 在有效行中存在缺失值")
                try:
                    init_values = init_series.astype(int).values
                except ValueError as e:
                    raise ValueError(f"初始列 '{init_col}' 包含非整数值: {str(e)}")
                if (init_values < 0).any():
                    raise ValueError("初始列包含负数，必须为0或正整数")

                cluster_ids = np.unique(init_values[init_values > 0])
                if len(cluster_ids) == 0:
                    raise ValueError("至少需要一个正整数作为聚类编号")
                expected_ids = np.arange(1, len(cluster_ids) + 1)
                if not np.array_equal(cluster_ids, expected_ids):
                    raise ValueError(f"聚类编号必须是从1开始的连续整数，发现编号: {cluster_ids}")

                k = len(cluster_ids)

                # 修改初始化质心部分代码
                centroids = []
                for cluster_id in cluster_ids:
                    cluster_mask = (init_values == cluster_id)
                    cluster_points = X_normalized[cluster_mask]
                    centroids.append(np.mean(cluster_points, axis=0))
                centroids_normalized = np.array(centroids)
                centroids_normalized, clusters_normalized = self.k_means_with_initial(X_normalized, centroids_normalized)


            # 运行K-means算法
            if self.cluster_var.isChecked():
                centroids_normalized, clusters_normalized = self.k_means(X_normalized, k)
            else:
                # 初始化质心逻辑
                centroids = []
                for cluster_id in cluster_ids:
                    cluster_mask = (init_values == cluster_id)
                    cluster_points = X_normalized[cluster_mask]
                    centroids.append(np.mean(cluster_points, axis=0))
                centroids_normalized = np.array(centroids)
                centroids_normalized, clusters_normalized = self.k_means_with_initial(X_normalized, centroids_normalized)

            # 添加维度校验
            if centroids_normalized.shape[0] != len(clusters_normalized):
                raise ValueError("质心数量与簇数量不匹配")
            # 轮廓系数计算
            labels = []
            valid_clusters = []
            for i, cluster in enumerate(clusters_normalized):
                if len(cluster) > 0:
                    labels.extend([i] * len(cluster))
                    valid_clusters.append(cluster)

            if len(valid_clusters) >= 2:
                silhouette_avg = silhouette_score(X_normalized, labels)
            else:
                silhouette_avg = -1  # 或提示无法计算

            # 计算结果指标
            cluster_metrics = self.calculate_cluster_metrics(X_normalized, clusters_normalized, centroids_normalized)
            overall_centroid = np.mean(X_normalized, axis=0)

            # 计算两种质心
            overall_centroid_original = original_means  # 原始数据总质心
            overall_centroid_normalized = np.mean(X_normalized, axis=0)  # 标准化后总质心

            analysis_method = "cluster_number" if self.cluster_var.isChecked() else "initial_column"

            # 转换原始坐标（如果需要）
            centroids_original = centroids_normalized * stds + means

            self.results = {
                "centroids_normalized": centroids_normalized,
                "centroids_original": centroids_original,
                "clusters": clusters_normalized,
                "silhouette_score": silhouette_avg,
                "cluster_metrics": cluster_metrics,
                "overall_centroid": overall_centroid,
                "overall_centroid_original": overall_centroid_original,
                "overall_centroid_normalized": overall_centroid_normalized,
                "analysis_method": analysis_method,  # 新增此行
                "centroid_distances": cdist(centroids_normalized, centroids_normalized, 'euclidean'),
                "is_standardized": self.std_checkbox.isChecked()
            }

            self.show_results()

        except Exception as e:
            QMessageBox.critical(self, "分析错误", f"聚类分析失败: {str(e)}")
    def k_means(self, data, k, max_iters=500, tol=1e-6):
        centroids = data[:k]
        for _ in range(max_iters):
            clusters = [[] for _ in range(k)]
            for x in data:
                distances = [np.linalg.norm(x - c) for c in centroids]
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(x)

            new_centroids = []
            for i, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    # 确保返回一维数组
                    mean_val = np.mean(cluster, axis=0)
                    if mean_val.ndim > 1:
                        mean_val = mean_val.ravel()
                    new_centroids.append(mean_val)
                else:
                    new_centroids.append(centroids[i])
            new_centroids = np.array(new_centroids)
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
        return centroids, clusters

    def k_means_with_initial(self, data, initial_centroids, max_iters=500, tol=1e-6):
        centroids = initial_centroids.copy()
        for _ in range(max_iters):
            clusters = [[] for _ in range(len(centroids))]
            for point in data:
                distances = [np.linalg.norm(point - c) for c in centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)

            new_centroids = []
            for i, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    # 保留原来的质心
                    new_centroids.append(centroids[i])
                else:
                    new_centroids.append(np.mean(cluster, axis=0))

            new_centroids = np.array(new_centroids)
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
        return centroids, clusters  # 同时返回质心和簇

    def silhouette_analysis(self, data, clusters):
        labels = np.zeros(data.shape[0], dtype=int)
        idx = 0
        for i, cluster in enumerate(clusters):
            for _ in cluster:
                labels[idx] = i
                idx += 1
        silhouette_avg = silhouette_score(data, labels)
        return silhouette_avg

    def calculate_cluster_metrics(self, data, clusters, centroids):
        metrics = []
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                metrics.append((0, 0.0, 0.0, 0.0))
            else:
                cluster_data = np.array(cluster)
                centroid = centroids[i]
                distances = np.linalg.norm(cluster_data - centroid, axis=1)
                wcss = np.sum(distances ** 2)
                avg_distance = np.mean(distances)
                max_distance = np.max(distances)
                metrics.append((len(cluster), wcss, avg_distance, max_distance))
        return metrics
    def show_results(self):
        result_text = "K 均值聚类分析:\n\n"
        result_text += f"聚类数: {len(self.results['centroids_normalized'])}\n"
        result_text += f"轮廓系数: {self.results['silhouette_score']:.4f}\n\n"

        result_text += "最终分割\n"
        result_text += "       观测值              到质心的  到质心的\n"
        result_text += "         个数  类内平方和  平均距离  最大距离\n"

        total_clusters = len(self.results['centroids_normalized'])
        for i in range(total_clusters):
            if i < len(self.results['cluster_metrics']):
                n, wcss, avg_dist, max_dist = self.results['cluster_metrics'][i]
            else:
                n, wcss, avg_dist, max_dist = 0, 0.0, 0.0, 0.0
            result_text += f"聚类{i + 1:<6} {n:<6} {wcss:<10.3f} {avg_dist:<8.3f} {max_dist:<8.3f}\n"

        result_text += "\n聚类质心\n变量   "
        for i in range(len(self.results["centroids_normalized"])):
            result_text += f"聚类{i + 1:<6}"
        result_text += "\n"

        for i, col_name in enumerate(self.selected_columns):
            result_text += f"{col_name:<6}"
            if self.results["is_standardized"]:
                for val in self.results["centroids_normalized"][:, i]:
                    result_text += f"{val:<8.4f}"
            else:
                for val in self.results["centroids_original"][:, i]:
                    result_text += f"{val:<8.4f}"
            result_text += "\n"

        if self.results["analysis_method"] == "initial_column":
            if self.results["is_standardized"]:
                result_text += "\n标准化总质心\n"
                for i, col_name in enumerate(self.selected_columns):
                    result_text += f"{col_name:<6}{self.results['overall_centroid_normalized'][i]:<8.4f}\n"
            else:
                result_text += "\n原始总质心\n"
                for i, col_name in enumerate(self.selected_columns):
                    result_text += f"{col_name:<6}{self.results['overall_centroid_original'][i]:<8.4f}\n"

        result_text += "\n聚类质心之间的距离\n        "
        centroid_dists = self.results["centroid_distances"]
        for i in range(len(centroid_dists)):
            result_text += f"聚类{i + 1:<6}"
        result_text += "\n"
        for i, row in enumerate(centroid_dists):
            result_text += f"聚类{i + 1:<6}"
            for val in row:
                result_text += f"{val:<8.4f}"
            result_text += "\n"

        # 使用 self.main_window.display_text() 显示结果
        self.main_window.display_text(result_text)
    def __init__(self, main_window=None):  # 添加参数
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("K均值聚类分析")
        self.data = None
        self.results = None
        self.selected_columns = []
        self.main_window = main_window  # 保存主窗口引用
        self.get_table_data()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Ui_MainWindow()
    main_window.show()
    sys.exit(app.exec())
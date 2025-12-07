import sys
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
from PyQt6 import QtWidgets, uic

from Algorithm import bat_max_flow as bat_maxflow
from Algorithm import CompleteMaxFlowGA as ga_maxflow
from Algorithm import run_edmond_karp_algo as edmonds_karp_maxflow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/traffic_optimizer.ui', self)

        self.layoutConvergence = QtWidgets.QVBoxLayout(self.layoutConvergence)
        self.layoutGraph = QtWidgets.QVBoxLayout(self.layoutGraph)
        self.layoutEsmond = QtWidgets.QVBoxLayout(self.layoutEsmond)

        # giữ dữ liệu đồ thị và kết quả vẽ
        self.graph_df = None
        self.edges = None
        self.capacities = None
        self.flows = None
        self.source = None
        self.sink = None


        # event nút
        self.btnLoadCSV.clicked.connect(self.load_csv)
        self.btnRun.clicked.connect(self.run_selected_algorithm)
        # Đặt giá trị mặc định GA
        self.spinpop_size.setValue(50)
        self.spingen_max.setValue(100)
        self.spinmuti_rate.setValue(0.1)
        self.spincross_rate.setValue(0.8)
        # Đặt giá trị mặc định BAT
        self.spinBat.setValue(3)
        self.spinLoops.setValue(1000)
        self.spinfmin.setValue(0.0)
        self.spinfmax.setValue(2.0)
        self.spinLoudmin.setValue(1.0)
        self.spinLoudmax.setValue(2.0)
        self.spinPratemin.setValue(0.0)
        self.spinPratemax.setValue(1.0)
        self.spinAlpha.setValue(0.9)
        self.spinGamma.setValue(0.9)

        #Đổi thuật toán 
        self.comboAlgorithm.currentTextChanged.connect(self.on_algorithm_changed)
    # Hàm load file CSV
    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn file CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.graph_df = pd.read_csv(path)
                # cập nhật label tên file (UI có lblFileName)
                try:
                    self.lblFileName.setText(f"{path.split('/')[-1]}")
                except Exception:
                    self.lblStatus.setText(f"Đã load: {path.split('/')[-1]}")
                self.lblStatus.setText("Đã load file mạng.")
            except Exception as e:
                self.lblStatus.setText("Lỗi đọc file")
                print("LOAD CSV ERROR:", e)

    #Hàm đổi thuộc tính
    def on_algorithm_changed(self):
        algo = self.comboAlgorithm.currentText()
        self.stackProperties.setCurrentIndex(0 if algo == "BAT" else 1)

    #Hàm chạy thuật toán đã chọn
    def run_selected_algorithm(self):
        algo = self.comboAlgorithm.currentText()

        # ánh xạ tên thuật toán → hàm tương ứng
        if algo == "BAT":
            self.run_bat()
        elif algo == "GA":
            self.run_ga()

        self.run_ek()
        self.graph()

    #Hàm vẽ graph của file
    def graph(self):

        if self.graph_df is None:
            self.lblStatus.setText("Chưa load đồ thị!")
            return

        # kiểm tra dữ liệu vẽ
        if not (self.edges and self.capacities and self.flows):
            self.lblStatus.setText("Chưa có kết quả để vẽ đồ thị.")
            return

        try:
            fig2 = Figure(figsize=(6, 5))
            canvas2 = FigureCanvas(fig2)
            ax2 = fig2.add_subplot(111)

            G = nx.DiGraph()

            # thêm node & edge cùng attribute flow/cap
            for (u, v), flow in self.flows.items():
                # nếu capacity có theo thứ tự edges
                try:
                    idx = self.edges.index((u, v))
                    cap = self.capacities[idx]
                except ValueError:
                    cap = None
                G.add_edge(u, v, flow=flow, cap=cap)

            # layout node
            nodes = list(G.nodes())

            source = self.source
            sink = self.sink

            if source not in nodes or sink not in nodes:
                pos = nx.spring_layout(G, seed=1)
            else:
                pos = {}
                pos[source] = (-1.0, 0.0)
                pos[sink] = (1.0, 0.0)
                middle_nodes = [n for n in nodes if n not in [source, sink]]
                if len(middle_nodes) == 0:
                    pass
                else:
                    step = 1.5 / (len(middle_nodes) + 1)
                    for i, n in enumerate(middle_nodes):
                        pos[n] = (-0.5 + step * (i + 1), -0.4 + 0.4 * (i % 3))

            # vẽ
            nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax2)
            nx.draw_networkx_labels(G, pos, ax=ax2)
            nx.draw_networkx_edges(G, pos, width=1.8, arrowsize=20, arrowstyle="-|>", ax=ax2)

            # edge label: flow/cap nếu có cap
            edge_label_dict = {}
            for u, v in G.edges():
                flow = G[u][v].get('flow', 0)
                cap = G[u][v].get('cap')
                edge_label_dict[(u, v)] = cap

            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, ax=ax2)

            ax2.set_title("Flow Graph")

            # xóa cũ → vẽ mới trong layoutGraph
            for i in reversed(range(self.layoutGraph.count())):
                widget = self.layoutGraph.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            self.layoutGraph.addWidget(canvas2)
            self.lblStatus.setText("Đã vẽ đồ thị.")
        except Exception as e:
            self.lblStatus.setText("Có lỗi trong quá trình vẽ đồ thị!")
            print("GRAPH ERROR:", e)

    #Hàm chạy và vẽ cho GA
    def run_ga(self):
        if self.graph_df is None:
            self.lblStatus.setText("Chưa load đồ thị!")
            return
        try:
            #Nhập vào các thuộc tính 
            self.textEdit.clear()

            source = self.txtSource.text().strip()
            sink = self.txtSink.text().strip()
            if source == "" or sink == "":
                self.lblStatus.setText("Vui lòng nhập Source và Sink.")
                return
            pop_size = self.spinpop_size.value()
            gen_max = self.spingen_max.value()
            muti_rate = self.spinmuti_rate.value()
            cross_rate = self.spincross_rate.value()

            self.lblStatus.setText(f"Đang chạy GA... ")
            QtWidgets.QApplication.processEvents()
            
            ga = ga_maxflow()

            #Hiển thị verbose
            old_stdout = sys.stdout
            sys.stdout = QTextEditLogger(self.textEdit)
            #Chạy GA
            all_paths_with_flows, best_flow_value,generation_history,history = ga.run_max_flow_ga(
                            df=self.graph_df,
                            source=source, 
                            sink=sink,
                            population_size=pop_size,
                            max_generations=gen_max,
                            mutation_rate=muti_rate, 
                            crossover_rate=cross_rate,
                            verbose=True)

            # Khôi phục stdout
            sys.stdout = old_stdout

            #Vẽ sơ đồ
            fig1 = Figure(figsize=(4, 3))
            canvas1 = FigureCanvas(fig1)
            ax1 = fig1.add_subplot(111)
            ax1.plot(history)
            ax1.set_title("GA Convergence")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Best Fitness")

            for i in reversed(range(self.layoutConvergence.count())):
                widget = self.layoutConvergence.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.layoutConvergence.addWidget(canvas1)
           
            edges = []
            capacities = []
            best_flow_value = {}

            for _, row in self.graph_df.iterrows():
                u = row['v_from']
                v = row['v_to']
                w = float(row['weight'])
                edges.append((u, v))
                capacities.append(w)
            flows = {}
            for e in edges:
                flows[e] = float(best_flow_value.get(e, 0.0))
            self.edges = edges
            self.capacities = capacities
            self.flows = flows
            self.source = source
            self.sink = sink
 
            try:
                self.lblStatus.setText(f"{best_flow_value:}")  
            except Exception:
                self.lblStatus.setText(f"GA hoàn tất! Best = {best_flow_value:} ")

            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.lblStatus.setText("Có lỗi trong quá trình chạy BAT!")
            print("GA ERROR:", e)

    #Hàm chạy và vẽ Bat
    def run_bat(self):
        if self.graph_df is None:
            self.lblStatus.setText("Chưa load đồ thị!")
            return
        ##Nhập vào các thuộc tính
        try:
            self.textEdit.clear()

            source = self.txtSource.text().strip()
            sink = self.txtSink.text().strip()
            if source == "" or sink == "":
                self.lblStatus.setText("Vui lòng nhập Source và Sink.")
                return

            num_bat = self.spinBat.value()
            max_loop = self.spinLoops.value()
            f_min = self.spinfmin.value()
            f_max = self.spinfmax.value()
            loud_min = self.spinLoudmin.value()
            loud_max = self.spinLoudmax.value()
            p_rate_min = self.spinPratemin.value()
            p_rate_max = self.spinPratemax.value()
            alpha = self.spinAlpha.value()
            gamma = self.spinGamma.value()

            self.lblStatus.setText(f"Đang chạy BAT... ")
            QtWidgets.QApplication.processEvents()

            #Hiển thị verbose
            old_stdout = sys.stdout
            sys.stdout = QTextEditLogger(self.textEdit)

            # chạy BAT 
            best_fit,best_paths_with_flow,best_edge_load,history = bat_maxflow(
                dataframe = self.graph_df, source=source, sink=sink,
                max_iterations=max_loop, n_bats=num_bat,
                f_min=f_min, f_max=f_max, A_min=loud_min, A_max=loud_max,
                r_min=p_rate_min, r_max=p_rate_max,
                alpha=alpha, gamma=gamma,
                max_paths=100, verbose=True
                )

            sys.stdout = old_stdout
            
            self.lblStatus.setText(str(best_edge_load))

            edges = []
            capacities = []
            for _, row in self.graph_df.iterrows():
                u = row['v_from']
                v = row['v_to']
                w = float(row['weight'])
                edges.append((u, v))
                capacities.append(w)
            flows = {}
            for e in edges:
                flows[e] = float(best_edge_load.get(e, 0.0))
            self.edges = edges
            self.capacities = capacities
            self.flows = flows
            self.source = source
            self.sink = sink

            # Vẽ biểu đồ
            fig1 = Figure(figsize=(4, 3))
            canvas1 = FigureCanvas(fig1)
            ax1 = fig1.add_subplot(111)
            ax1.plot(history)
            ax1.set_title("Bat Convergence")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best Fitness")

            # Xóa nội dung cũ và gắn nội dung mới
            for i in reversed(range(self.layoutConvergence.count())):
                widget = self.layoutConvergence.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.layoutConvergence.addWidget(canvas1)

            # Cập nhật status 
            try:
                self.lblStatus.setText(f"{best_fit:.6f}")  
            except Exception:
                self.lblStatus.setText(f"BAT hoàn tất! Best = {best_fit:.3f} ")

            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.lblStatus.setText("Có lỗi trong quá trình chạy BAT!")
            print("BAT ERROR:", e)

    #Hàm chạy cho EK
    def run_ek(self):
        if self.graph_df is None:
            self.lblStatus.setText("Chưa load đồ thị!")
            return
        try:
            source = self.txtSource.text().strip()
            sink = self.txtSink.text().strip()
            if source == "" or sink == "":
                self.lblStatus.setText("Vui lòng nhập Source và Sink.")
                return

            maxflow, residual, history = edmonds_karp_maxflow(
                dataframe=self.graph_df,
                source=source,
                sink=sink,
                verbose=False
            )
            # hiển thị max flow 
            try:
                self.txtMaxFlowEK.setText(f"Max flow: {str(maxflow)}")
            except Exception:
                self.lblStatus.setText(f"EK MaxFlow = {maxflow}")

            # VẼ biểu đồ tiến trình Edmonds-Karp
            for i in reversed(range(self.layoutEsmond.count())):
                widget = self.layoutEsmond.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            fig = Figure(figsize=(4, 3))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.plot(history, linewidth=2)
            ax.set_title("Edmonds-Karp Convergence")
            ax.set_xlabel("Augment Step")
            ax.set_ylabel("Current Max Flow")

            self.layoutEsmond.addWidget(canvas)

            self.lblStatus.setText(f"Edmonds–Karp hoàn tất! MaxFlow = {maxflow}")

        except Exception as e:
            self.lblStatus.setText("Lỗi khi chạy Edmonds-Karp!")
            print("EK ERROR:", e)
        
    def run_application():
        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        win.show()
        sys.exit(app.exec())

#Class dùng để chuyển print từ terminal lên textedit của verbose
class QTextEditLogger:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message):
        if message.strip():  # Bỏ dòng trống
            self.text_edit.append(message)

    def flush(self):
        pass  # Không cần dùng nhưng phải có


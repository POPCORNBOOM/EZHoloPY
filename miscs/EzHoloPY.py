import numpy as np
from stl import mesh
import copy
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QWidget, QLabel, QSlider
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *

class STLViewer(QGLWidget):
    def __init__(self, parent=None):
        super(STLViewer, self).__init__(parent)
        self.mesh = None
        self.angle_x = 0
        self.angle_y = 0
        self.zoom = 1.0
        self.b = 1  # 双曲线的参数
        self.r = 10
        self.transformed_points = []
        self.isMoving = False
        self.last_mouse_pos = None
        self.show_model = True
        self.model_depth = 0  # Depth of the model relative to the panel
        self.interpolation_density = 1  # Interpolation density
        self.show_plane = False  # Toggle for displaying the scratch plane

    def load_stl(self, filepath):
        self.mesh = mesh.Mesh.from_file(filepath)
        vertices = self.mesh.vectors.reshape(-1, 3)

        # Calculate the scale factor to normalize the XZ longest edge to 50
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        size_xz = max(max_bounds[0] - min_bounds[0], max_bounds[2] - min_bounds[2])
        scale_factor = 50.0 / size_xz

        # Apply scaling to the mesh
        self.mesh.vectors *= scale_factor
        vertices = self.mesh.vectors.reshape(-1, 3)

        # Recalculate model center and bounds
        self.model_center = np.mean(vertices, axis=0)
        self.model_bounds = np.max(np.ptp(vertices, axis=0)) / 2.0  # Half of the max span
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glOrtho(-50 / self.zoom, 50 / self.zoom, -50 / self.zoom, 50 / self.zoom, -500, 500)

        
        # 固定屏幕平面上，为每个点绘制双曲线
        if self.mesh:
            vertices = self.mesh.vectors.reshape(-1, 3)
            #self.draw_hyperbolas_for_points(vertices)
            # 绘制模型
            if(self.show_plane):
                self.draw_plane()
            if self.show_model:

                glPushMatrix()
                glTranslatef(0, 0, self.model_depth)
                glRotatef(self.angle_x, 1, 0, 0)
                glRotatef(self.angle_y, 0, 1, 0)

                model_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
                self.transformed_points = copy.deepcopy(self.get_transformed_points(model_matrix))

                self.draw_model()
                self.draw_points()
                glPopMatrix()
                self.draw_hyperbolas_for_points(self.transformed_points)





    def draw_plane(self):
        glColor3f(0.0, 0.0, 1.0)  # Blue for the scratch plane
        glBegin(GL_QUADS)
        glVertex3f(-50, -50, 0)
        glVertex3f(50, -50, 0)
        glVertex3f(50, 50, 0)
        glVertex3f(-50, 50, 0)
        glEnd()

    def draw_hyperbolas_for_points(self, vertices):
        """
        为每个点在固定屏幕平面上绘制双曲线。
        :param vertices: 模型的顶点列表
        """
        glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 0.0)  # 黄色双曲线
        step = 0.1  # 双曲线的精度控制

        for point in vertices:
            m, n, l = point  # 提取点的坐标
            sign_l = 1 if l > 0 else -1  # 决定双曲线的上下支
            #print(m, n, l)

            glBegin(GL_LINE_STRIP)
            for x in np.arange(-self.r, self.r, step):  # x 范围 [-r, r]
                try:
                    # 计算双曲线 y 坐标
                    y = n + sign_l + np.sqrt((x - m) ** 2 / self.b ** 2 + 1)
                    glVertex2f(x, y)  # 添加顶点到绘制中
                except ValueError:
                    pass  # 忽略无效的计算值（如负数开方）
            glEnd()
        glEnable(GL_DEPTH_TEST)

    def get_transformed_points(self, model_matrix):
        transformed_points = []
        vertices = self.mesh.vectors.reshape(-1, 3)

        # 将每个点与变换矩阵相乘
        for point in vertices:
            # 将点扩展为齐次坐标 (x, y, z, 1)
            homogeneous_point = np.array([point[0], point[1], point[2], 1.0])
            
            # 将变换矩阵应用于点
            transformed_point = np.dot(model_matrix, homogeneous_point)
            
            # 将齐次坐标转换为非齐次坐标
            transformed_points.append(transformed_point[:3])  # 只取x, y, z

        return transformed_points


    def draw_model(self):
        glColor3f(0.5, 0.5, 0.5)  # Gray for the model
        glBegin(GL_TRIANGLES)
        for triangle in self.mesh.vectors:
            for vertex in triangle:
                glVertex3f(*vertex)
        glEnd()

    def draw_points(self):
        vertices = self.mesh.vectors.reshape(-1, 3)
        glColor3f(1, 0, 0)  # Red for points
        glPointSize(5.0)  # Set point size
        glBegin(GL_POINTS)
        for point in vertices:
            glVertex3f(*point)
            self.draw_hyperbola(*point)  # Draw hyperbola for each point
        glEnd()


    def draw_points(self):
        """在屏幕上绘制点并获取其屏幕坐标"""
        vertices = self.mesh.vectors.reshape(-1, 3)
        glColor3f(1, 0, 0)  # 红色点
        glPointSize(5.0)  # 设置点的大小
        glBegin(GL_POINTS)
        for point in vertices:
            glVertex3f(*point)
        glEnd()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect_ratio = w / h
        glOrtho(-50 * aspect_ratio / self.zoom, 50 * aspect_ratio / self.zoom, -50 / self.zoom, 50 / self.zoom, -500, 500)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()
        if event.button() == Qt.MiddleButton:
            self.isMoving = True

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MiddleButton:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.angle_x -= dy * 0.5
            self.angle_y -= dx * 0.5
        self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.isMoving = False

    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() / 1200.0  # Adjust zoom to control distance
        self.zoom = max(0.6, min(10, self.zoom))
        self.resizeGL(self.width(), self.height())  # Update projection with new zoom
        self.update()

    def set_model_depth(self, value):
        self.model_depth = value
        self.update()

    def toggle_plane(self):
        self.show_plane = not self.show_plane
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.viewer = STLViewer(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Orthographic STL Viewer')
        self.resize(1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.viewer, stretch=5)  # Ensure viewer takes most of the space

        control_layout = QVBoxLayout()

        toggle_button = QPushButton("Toggle Model Display")
        toggle_button.clicked.connect(self.toggle_model_display)
        control_layout.addWidget(toggle_button)

        plane_button = QPushButton("Toggle Scratch Plane")
        plane_button.clicked.connect(self.viewer.toggle_plane)
        control_layout.addWidget(plane_button)

        depth_slider = QSlider(Qt.Horizontal)
        depth_slider.setRange(-100, 100)
        depth_slider.setValue(0)
        depth_slider.valueChanged.connect(self.viewer.set_model_depth)
        control_layout.addWidget(QLabel("Adjust Model Depth"))
        control_layout.addWidget(depth_slider)

        layout.addLayout(control_layout, stretch=1)

        self.menu = self.menuBar().addMenu('File')
        open_action = self.menu.addAction('Open STL')
        open_action.triggered.connect(self.open_stl_file)

    def toggle_model_display(self):
        self.viewer.show_model = not self.viewer.show_model
        self.viewer.update()

    def open_stl_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open STL File', '', 'STL Files (*.stl)', options=options)
        if filepath:
            self.viewer.load_stl(filepath)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

class MyGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(MyGLWidget, self).__init__(parent)
        self.setMinimumSize(100, 100)
        self.zoom = 1.0

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect_ratio = w / h
        glOrtho(-50 * aspect_ratio / self.zoom, 50 * aspect_ratio / self.zoom, -50 / self.zoom, 50 / self.zoom, -500, 500)
        print(aspect_ratio)
        print(self.zoom)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 清空颜色和深度缓冲区
        glLoadIdentity()  # 重置模型视图矩阵
        glOrtho(-50 / self.zoom, 50 / self.zoom, -50 / self.zoom, 50 / self.zoom, -500, 500)           

        glColor3f(0.0, 0.0, 1.0)  # Blue for the scratch plane
        glBegin(GL_QUADS)
        glVertex3f(-50, -50, 0)
        glVertex3f(50, -50, 0)
        glVertex3f(50, 50, 0)
        glVertex3f(-50, 50, 0)
        glEnd()


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('OpenGL Plane Example')
        self.setGeometry(100, 100, 800, 600)

        # Create the OpenGL widget
        self.glWidget = MyGLWidget(self)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.glWidget)

        # Set the main widget
        mainWidget = QWidget(self)
        mainWidget.setLayout(layout)
        self.setCentralWidget(mainWidget)

    def scrollEvent(self, event):
        if event.angleDelta().y() > 0:
            self.glWidget.zoom *= 1.1
        else:
            self.glWidget.zoom /= 1.1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def init_opengl():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

def build_transformation_matrix_opengl():
    glPushMatrix()
    glLoadIdentity()
    
    glRotatef(45, 0, 0, 1)  # 绕Z轴旋转45度
    glScalef(2.0, 2.0, 2.0)  # 在x、y、z轴上均匀缩放

    matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    glPopMatrix()
    return matrix

import torch
import numpy as np

def build_transformation_matrix_torch():
    # 设置旋转角度并创建旋转矩阵
    angle = torch.tensor([45 * np.pi / 180])  # 转换为弧度
    R = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle),  torch.cos(angle), 0],
        [0,                0,                1]
    ])

    # 创建缩放矩阵
    S = torch.diag(torch.tensor([2.0, 2.0, 2.0]))

    # 应用旋转和缩放
    M = torch.mm(R, S)  # 注意顺序，确保与 OpenGL 一致

    return M


def main():
    init_opengl()
    matrix = build_transformation_matrix_opengl()
    print("OpenGL Transformation Matrix:")
    print(matrix)
    pygame.quit()
    
    # 
    pytorch_matrix = build_transformation_matrix_torch()
    print("PyTorch Transformation Matrix:")
    print(pytorch_matrix)

if __name__ == "__main__":
    main()



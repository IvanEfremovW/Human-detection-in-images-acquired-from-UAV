from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import OpenGL.constant

import numpy as np
from numpy.typing import NDArray

class Shader:
    def __init__(self, shader_type: OpenGL.constant.IntConstant, filename: str):
        self.id = self._load_shader(shader_type, filename)

    def _load_shader(self, shader_type, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            shader_code = file.read()

        shader = compileShader(shader_code, shader_type)
        
        return shader
    
    def delete(self):
        glDeleteShader(self.id)
   

class Program:
    def __init__(self, *shaders: OpenGL.constant.IntConstant):
        self.id = self._create_program(*shaders)

    def _create_program(self, *shaders):
        shader_ids = [shader.id for shader in shaders]
        program = compileProgram(*shader_ids)
        
        return program
        
    def set_uniform1i(self, name: str, value: int):
        location = glGetUniformLocation(self.id, name)
        glUniform1i(location, value)
    
    def set_uniform1f(self, name: str, value: int):
        location = glGetUniformLocation(self.id, name)
        glUniform1f(location, float(value))
    
    def use(self):
        glUseProgram(self.id)
        
    def delete(self):
        glDeleteProgram(self.id)   


class Buffer:
    def __init__(self, target: OpenGL.constant.IntConstant):
        self.id = glGenBuffers(1)
        self.target = target

    def bind(self):
        glBindBuffer(self.target, self.id)
    
    def bind_base(self, index: int):
        glBindBufferBase(self.target, index, self.id)
    
    def data(self, data: NDArray[np.float32], usage: OpenGL.constant.IntConstant):
        glBufferData(self.target, data.nbytes, data, usage)

    def unbind(self):
        glBindBuffer(self.target, 0)
    
    def delete(self):
        glDeleteBuffers(1, [self.id])


class VertexArray:
    def __init__(self):
        self.id = glGenVertexArrays(1)
        
    def bind(self):
        glBindVertexArray(self.id)

    def unbind(self):
        glBindVertexArray(0)
    
    def linkAttrib(
        self, 
        layout: int, 
        size: int, 
        type: OpenGL.constant.IntConstant, 
        normalized: OpenGL.constant.IntConstant, 
        stride: int,
        offset: int
    ): 
        glVertexAttribPointer(layout, size, type, normalized, stride, GLvoidp(offset))
        glEnableVertexAttribArray(layout)

    def delete(self):
        glDeleteVertexArrays(1, [self.id])


class Texture:
    def __init__(self):
        self.id = glGenTextures(1)
        
        self.bind()
        self._set_parameters()
        self.unbind()
        
    def bind(self, unit:int=0):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.id)

    def bind_image_texture(self, unit: int, access: OpenGL.constant.IntConstant):
        glBindImageTexture(unit, self.id, 0, GL_FALSE, 0, access, GL_RGBA8)

    def unbind(self):
        glBindTexture(GL_TEXTURE_2D, self.id)

    def _set_parameters(self):
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def load_data(self, width: int,
                  height: int,
                  data: NDArray[np.uint8],
                  format:OpenGL.constant.IntConstant=GL_RGBA
    ):
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data)
    
    def delete(self):
        glDeleteTextures(1, (self.id,))
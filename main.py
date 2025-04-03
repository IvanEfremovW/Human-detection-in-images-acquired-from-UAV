from PIL import Image

import numpy as np
from OpenGL.GL import *
import glfw




class Shader:
    def __init__(self, shader_type, filename):
        self.id = self._load_shader(shader_type, filename)

    def _load_shader(self, shader_type, filename):
        with open(filename, 'r') as file:
            shader_code = file.read()
            
        shader = glCreateShader(shader_type)
        
        glShaderSource(shader, shader_code)
        glCompileShader(shader)
        
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(shader)
            raise RuntimeError(f"Shader compilation failed: {info_log.decode()}")
        
        return shader
    
    def delete(self):
        glDeleteShader(self.id)
   

class Program:
    def __init__(self, vertex_shader, fragment_shader):
        self.id = self._create_program(vertex_shader.id, fragment_shader.id)
    
    def _create_program(self, vertex_shader_id, fragment_shader_id):
        program = glCreateProgram()

        glAttachShader(program, vertex_shader_id)
        glAttachShader(program, fragment_shader_id)
        
        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            info_log = glGetProgramInfoLog(program)
            raise RuntimeError(f"Program linking failed: {info_log.decode()}")
        
        return program
    
    def use(self):
        glUseProgram(self.id)
        
    def delete(self):
        glDeleteProgram(self.id)   


class Buffer:
    def __init__(self, target):
        self.id = glGenBuffers(1)
        self.target = target

    def bind(self):
        glBindBuffer(self.target, self.id)
    
    def data(self, data, usage):
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
    
    def linkAttrib(self, layout, size, type, normalized, stride, offset):
        glVertexAttribPointer(layout, size, type, normalized, stride, GLvoidp(offset))
        glEnableVertexAttribArray(layout)
    
    """
    def enable_attrib_array(self, index):
        glEnableVertexAttribArray(index)
     
    def vertex_attrib_pointer(self, index, size, type, normalized, stride, offset):
        glVertexAttribPointer(index, size, type, normalized, stride, GLvoidp(offset))
    """

    def delete(self):
        glDeleteVertexArrays(1, [self.id])


class Texture:
    def __init__(self):
        self.id = glGenTextures(1)
        
        self.bind()
        self._set_parameters()
        self.unbind()
        
    def bind(self, unit=0):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.id)

    def bind_image_texture(self, unit, access):
        glBindImageTexture(unit, self.id, 0, GL_FALSE, 0, access, GL_RGBA32F)

        
    def unbind(self):
        glBindTexture(GL_TEXTURE_2D, self.id)

    def _set_parameters(self):
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def load_data(self, width, height, data, format=GL_RGBA):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, format, GL_UNSIGNED_BYTE, data)
    
    def delete(self):
        glDeleteTextures(1, [self.id])
       

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.width, self.height, self.image_data = self._load_image()
        
        self._create_context()
        
        self.vertices = np.array([
            -1.0, -1.0, 0.0,   0.0, 0.0,
             1.0, -1.0, 0.0,   1.0, 0.0,
             1.0,  1.0, 0.0,   1.0, 1.0,
            -1.0,  1.0, 0.0,   0.0, 1.0
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)
        
        self.vertex_shader = Shader(GL_VERTEX_SHADER, 'shaders/vertex_shader.glsl')
        self.fragment_shader = Shader(GL_FRAGMENT_SHADER, 'shaders/fragment_shader.glsl')

        self.program = Program(self.vertex_shader, self.fragment_shader)
        
        self.vao = VertexArray()
        self.vbo = Buffer(GL_ARRAY_BUFFER)
        self.ebo = Buffer(GL_ELEMENT_ARRAY_BUFFER)

        self.vao.bind()
        self.vbo.bind()
        self.ebo.bind()
        
        self.vbo.data(self.vertices, GL_STATIC_DRAW)
        self.ebo.data(self.indices, GL_STATIC_DRAW)
        
        self.vao.linkAttrib(0, 3, GL_FLOAT, GL_FALSE, 5 * self.vertices.itemsize, 0)
        self.vao.linkAttrib(1, 2, GL_FLOAT, GL_FALSE, 5 * self.vertices.itemsize, 3 * self.vertices.itemsize)
        
        self.vao.unbind()
        self.vbo.unbind()
        self.ebo.unbind()  
        
        self.texture = Texture()
        
        self.texture.bind()
        self.texture.load_data(self.width, self.height, self.image_data)
        self.texture.unbind()
        
    def _create_context(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(self.width, self.height, "Image Processing", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
        
        glfw.make_context_current(self.window)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
    def _load_image(self):
        image = Image.open(self.image_path).convert("RGBA")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        width, height = image.size
        
        image_data = np.array(image.getdata(), np.uint8).reshape(height, width, 4) # Ensure RGBA

        return width, height, image_data
    
    def draw(self):
        self.program.use()
        
        self.texture.bind()
        self.vao.bind()
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        self.vao.unbind()
        
    def render_window(self):
        while not glfw.window_should_close(self.window):
            glClearColor(0.07, 0.13, 0.17, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            self.draw()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        self.delete()
        
    def delete(self):
        self.vao.delete()
        self.vbo.delete()
        self.ebo.delete()
        
        self.program.delete()
        self.vertex_shader.delete()
        self.fragment_shader.delete()
        
        glfw.destroy_window(self.window)
        glfw.terminate()
    
if __name__ == "__main__":
    processor = ImageProcessor('images/image.jpg')
    processor.render_window()
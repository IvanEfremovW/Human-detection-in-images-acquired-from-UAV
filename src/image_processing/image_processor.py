from PIL import Image
import numpy as np
from OpenGL.GL import *
import glfw

from image_processing.opengl_utils import Shader, Program, Buffer, VertexArray, Texture

class ImageProcessor:
    def __init__(self, image_path, tile_size=16, clip_limit=5.0):
        self.image_path = image_path
        self.width, self.height, self.image_data = self._load_image()
        
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        
        self.NUM_BINS = 101
        
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
        
        self.program.use()
        
        self.program.set_uniform1i('u_tileSize', self.tile_size)
        self.program.set_uniform1i('u_numBins', self.NUM_BINS)
        self.program.set_uniform1f('u_clipLimit', self.clip_limit)
        
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

        self.program.use()
        
        
    def _create_context(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(self.width, self.height, "Image Processing", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        glfw.window_hint(glfw.VISIBLE, False)
                
        glfw.make_context_current(self.window)
        
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
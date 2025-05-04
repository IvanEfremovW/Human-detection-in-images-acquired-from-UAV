from PIL import Image
import numpy as np
from OpenGL.GL import *
import glfw

from image_processing.opengl_utils import Shader, Program, Buffer, VertexArray, Texture

from time import perf_counter

class ImageProcessor:
    def __init__(self, tile_size=8, clip_limit=2.0):
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        
        self.image_width = 0
        self.image_height = 0
        self.texture = None
        
        self.window = None
        
        self._create_context()
        self._init_opengl_resources()

        self.program.use()
        
    def _create_context(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        glfw.window_hint(glfw.VISIBLE, False)
        
        self.window = glfw.create_window(2048, 2048, "Image Processing", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

                
        glfw.make_context_current(self.window)

    def _init_opengl_resources(self):         
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
        
        self.vertex_shader = Shader(GL_VERTEX_SHADER, 'src/image_processing/shaders/vertex_shader.glsl')
        self.fragment_shader = Shader(GL_FRAGMENT_SHADER, 'src/image_processing/shaders/fragment_shader.glsl')

        self.program = Program(self.vertex_shader, self.fragment_shader)
        
        self.program.use()
        
        self.program.set_uniform1i('u_tileSize', self.tile_size)
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
    
    def process(self, input_path, output_path):
        
        process_time = perf_counter()
        
        if not self._load_image(input_path):
            print("Failed to load image.")
            return False
        
        load_time = 1000 * (perf_counter() - process_time)
        
        shader_init_time = perf_counter()
        
        self.texture = Texture()
        
        self.texture.bind()
        self.texture.load_data(self.image_width, self.image_height, self.image_data)
        self.texture.unbind()

        shader_init_time = 1000 * (perf_counter() - shader_init_time)
        
        shader_processing_time = perf_counter()
        
        self._draw()

        shader_processing_time = 1000 * (perf_counter() - shader_processing_time)
        
        saving_time = perf_counter()
        
        self._render_to_file(output_path)
        
        saving_time = 1000 * (perf_counter() - saving_time)

        self.texture.delete()
        
        process_time = 1000 * (perf_counter() - process_time)

        perf_log = {
            'time_total': process_time,
            'time_shader_init': shader_init_time,
            'time_shader_processing': shader_processing_time,
            'time_loading_image': load_time,
            'time_saving_image': saving_time
            }

        return perf_log
        
    def _load_image(self, image_path):
        
        self.image_path = image_path
        
        try:
            image = Image.open(self.image_path)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
            self.image_width, self.image_height = image.size

            self.image_data = np.array(image.convert("RGBA"), dtype=np.uint8)
        except FileNotFoundError:
            print(f"Error: File '{image_path}' not found.")
            return False
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
        return True
    
    def _draw(self):
        glClearColor(0.07, 0.13, 0.17, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
        glViewport(0, 0, self.image_width, self.image_height)
         
        self.program.use()
        
        self.texture.bind()
        self.vao.bind()
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        self.vao.unbind()
    
    def _render_to_file(self, output_path):

        pixels = glReadPixels(0, 0, self.image_width, self.image_height, GL_RGBA, GL_UNSIGNED_BYTE)

        image = Image.frombuffer("RGBA", (self.image_width, self.image_height), pixels, "raw", "RGBA", 0, 0)
        
        image = image.convert("RGB")
        image.save(output_path, "JPEG")

    def delete(self):
        self.vao.delete()
        self.vbo.delete()
        self.ebo.delete()
        
        self.program.delete()
        self.vertex_shader.delete()
        self.fragment_shader.delete()
        
        glfw.destroy_window(self.window)
        glfw.terminate()
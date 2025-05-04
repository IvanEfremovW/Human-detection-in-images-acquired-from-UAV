from OpenGL.GL import *
import glfw
import numpy as np
from PIL import Image

from image_processing.opengl_utils import *

from time import perf_counter

class ImageProcessor:
    def __init__(self, tile_size=8, clip_limit=2.0):
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        
        self.image_width = 0
        self.image_height = 0
        self.image_data = None
        
        self.input_texture = None
        self.output_texture = None
        
        self.histogram_cdf_buffer = None
        self.histogram_cdf_size = 0
        
        self.cdf_program = None
        self.clahe_program = None
        
        self.window = None

        self._create_context()
        self._init_opengl_resourses()
        
    def _create_context(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(1, 1, "Image Processing", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        glfw.window_hint(glfw.VISIBLE, False)
                
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        
    def _init_opengl_resourses(self):
        self.cdf_shader = Shader(GL_COMPUTE_SHADER, 'src/image_processing/shaders/cs_cdf.glsl')
        self.clahe_shader = Shader(GL_COMPUTE_SHADER, 'src/image_processing/shaders/cs_clahe.glsl')
        
        self.cdf_program = Program(self.cdf_shader)
        self.clahe_program = Program(self.clahe_shader)
        
        if self.cdf_program is None or self.clahe_program is None:
            raise RuntimeError('Failed to create program')
    
    
    def process(self, image_path, output_path):
        
        process_time = perf_counter()
        
        if not self._load_image(image_path):
            print("Failed to load image.")
            return False

        load_time = 1000 * (perf_counter() - process_time)
        
        shader_init_time = perf_counter()
        
        self._create_textures_and_buffers()

        shader_init_time = 1000 * (perf_counter() - shader_init_time)
        
        try:
            shader_processing_time = perf_counter()
            
            self._compute_clahe()
            
            shader_processing_time = 1000 * (perf_counter() - shader_processing_time)
            
            saving_time = perf_counter()
            
            self._save_image(output_path)
            
            saving_time = 1000 * (perf_counter() - saving_time)
            
            process_time = 1000 * (perf_counter() - process_time)
            
            perf_log = {
                'time_total': process_time,
                'time_shader_init': shader_init_time,
                'time_shader_processing': shader_processing_time,
                'time_loading_image': load_time,
                'time_saving_image': saving_time
                }
            
            return perf_log
        
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
        finally:
            self._cleanup_image_resources()
            
              
    def _load_image(self, image_path):
        self.image_path = image_path
        
        try:
            img = Image.open(self.image_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            self.image_width, self.image_height = img.size
            self.image_data = np.array(img.convert("RGBA"), dtype=np.uint8)
        except FileNotFoundError:
            print(f"Error: File '{image_path}' not found.")
            return False
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
        return True
    
    def _create_textures_and_buffers(self):
        self.input_texture = Texture()
        self.input_texture.bind()
        self.input_texture.load_data(self.image_width, self.image_height, self.image_data)
        self.input_texture.unbind()
        
        self.output_texture = Texture()
        self.output_texture.bind()
        self.output_texture.load_data(self.image_width, self.image_height, None)
        self.output_texture.unbind()
        
        # Histogram/CDF buffer setup
        num_tiles_x = (self.image_width + self.tile_size - 1) // self.tile_size
        num_tiles_y = (self.image_height + self.tile_size - 1) // self.tile_size
        num_tiles = num_tiles_x * num_tiles_y
        num_bins = 101
        
        tile_data_size = (num_bins + num_bins) * 4  #  (histogram + cdf) * float (4 bytes)
        
        self.histogram_cdf_size = num_tiles * tile_data_size

        self.histogram_cdf_buffer = Buffer(GL_SHADER_STORAGE_BUFFER)
        self.histogram_cdf_buffer.bind()
        
        data = np.zeros(self.histogram_cdf_size // 4, dtype=np.float32)
        self.histogram_cdf_buffer.data(data, GL_DYNAMIC_COPY)
        self.histogram_cdf_buffer.bind_base(0)
        self.histogram_cdf_buffer.unbind()
        
    def _compute_clahe(self):
        
        # Create CDF map in buffer
        self.cdf_program.use()

        self.cdf_program.set_uniform1i("u_tileSize", self.tile_size)
        self.cdf_program.set_uniform1i("imageWidth", self.image_width)
        self.cdf_program.set_uniform1i("imageHeight", self.image_height)
        self.cdf_program.set_uniform1f("u_clipLimit", self.clip_limit)
        
        self.input_texture.bind_image_texture(1, GL_READ_ONLY)

        num_groups_x = (self.image_width + self.tile_size - 1) // self.tile_size
        num_groups_y = (self.image_height + self.tile_size - 1) // self.tile_size

        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Set new values to pixels
        self.clahe_program.use()

        self.clahe_program.set_uniform1i("u_tileSize", self.tile_size)
        self.clahe_program.set_uniform1i("imageWidth", self.image_width)
        self.clahe_program.set_uniform1i("imageHeight", self.image_height)

        self.input_texture.bind_image_texture(1, GL_READ_ONLY)
        self.output_texture.bind_image_texture(2, GL_WRITE_ONLY)
        
        glDispatchCompute(num_groups_x, num_groups_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        glUseProgram(0)

    def _save_image(self, output_path):

        self.output_texture.bind()
        
        pixels = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombuffer("RGBA", (self.image_width, self.image_height), pixels, "raw", "RGBA", 0, 0)

        try:
            image = image.convert("RGB")
            image.save(output_path, "JPEG")
        except Exception as e:
            print(f"Error saving image: {e}")

    def _cleanup_image_resources(self):
        if self.input_texture:
            self.input_texture.delete()
            self.input_texture = None
        if self.output_texture:
            self.output_texture.delete()
            self.output_texture = None
        if self.histogram_cdf_buffer:
            self.histogram_cdf_buffer.delete()
            self.histogram_cdf_buffer = None
            
    def delete(self):
        self.cdf_shader.delete()
        self.clahe_shader.delete()
        
        self.cdf_program.delete()
        self.clahe_program.delete()
        
        glfw.destroy_window(self.window)
        glfw.terminate()
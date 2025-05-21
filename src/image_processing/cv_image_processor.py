import cv2
from PIL import Image
import numpy as np
from time import perf_counter

class OpenCVprocessor:
    def __init__(self, tile_size=8, clip_limit=2.0):
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))

    def process(self, image_path, output_path):

        process_time = perf_counter()

        load_time = perf_counter()
        
        img = self._load_image(image_path)
        
        load_time = 1000 * (perf_counter() - load_time)

        shader_processing_time = perf_counter()

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        cl = self.clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        final_data = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        shader_processing_time = 1000 * (perf_counter() - shader_processing_time)

        saving_time = perf_counter()
        
        self._save_image(final_data, output_path)
        
        saving_time = 1000 * (perf_counter() - saving_time)

        process_time = 1000 * (perf_counter() - process_time)

        perf_log = {
            'time_total': process_time,
            'time_shader_init': 0,
            'time_shader_processing': shader_processing_time,
            'time_loading_image': load_time,
            'time_saving_image': saving_time
        }

        return perf_log
    
    def _load_image(image_path):
        try:
            img_pil = Image.open(image_path).convert("RGB") 
            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            return img
            
        except Exception as e:
            print(f"Error loading image with PIL: {e}")
            
            return None
                
    def _save_image(data, output_path):
        try:
            final_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(final_rgb)
            img_pil = img_pil.convert("RGB")
            img_pil.save(output_path, "JPEG")
        except Exception as e:
            print(f"Error saving image: {e}")
        except Exception as e:
            print(f"Error saving image with PIL: {e}")

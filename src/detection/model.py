from ultralytics import YOLO
from torch import device
from torch.cuda import is_available
import numpy as np
from PIL import Image

class Model():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = device('cuda' if is_available() else 'cpu')

    def predict(self, input_path, output_path=None):
        results = self.model.predict(input_path, device=self.device, verbose=False)

        if output_path is not None:
            for result in results:
                result.save(output_path)
        
        return results
        
    def validate(self, validation_set_yaml):
        self.model.val(data=validation_set_yaml)
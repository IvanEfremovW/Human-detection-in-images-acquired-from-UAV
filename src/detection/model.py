from ultralytics import YOLO
from torch import device
from torch.cuda import is_available

class Model():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = device('cuda' if is_available() else 'cpu')

    def predict(self, input_path, output_path):
        results = self.model.predict(input_path, device=self.device)

        for result in results:
            result.save(output_path)
    
    def validate(self, validation_set_yaml):
        self.model.val(data=validation_set_yaml)
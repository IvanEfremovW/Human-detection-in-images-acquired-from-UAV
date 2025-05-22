import argparse

from detection.model import Model
from image_processing.image_processor import ShaderProcessor

def main():
    parser = argparse.ArgumentParser('Process image and detect objects')
    
    parser = argparse.ArgumentParser(description="Process image with CLAHE and object detection")
    parser.add_argument("-i", "--input_path", required=True, help="Path to the input image")
    parser.add_argument("-p", "--processing_result_path", required=True, help="Path to save the CLAHE processed image")
    parser.add_argument("-d", "--detection_result_path", required=True, help="Path to save the image with object detection results")
    
    args = parser.parse_args()
    
    model = Model('src/detection/models/YOLOv11_cl2_tile16.pt')
    processor = ShaderProcessor()
    
    processor.process(args.input_path, args.processing_result_path)

    model.predict(args.processing_result_path, args.detection_result_path)
    
if __name__ == '__main__':
    main()
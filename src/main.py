from image_processing.image_processor import ImageProcessor

def main():
    processor = ImageProcessor()
    processor.process('input.png','output.png')

if __name__ == "__main__":
    main()
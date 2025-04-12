from image_processing.image_processor import ImageProcessor

def main():
    processor = ImageProcessor('images/image.jpg')
    processor.render_window()

if __name__ == "__main__":
    main()
import cv2
import numpy as np

from heart_analyzer import HeartAnalyzer


def upload_img():
    images = []
    for i in range(1, 6):
        img_path = f"in_img/hearts {i}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images



def main():
    images = upload_img()
    analyzer = HeartAnalyzer()
    # analyzer.clean_images(images)
    analyzer.detect_hearts()



if __name__ == "__main__":
    main()

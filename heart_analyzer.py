import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_image(image, text="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)
    plt.show()


class HeartAnalyzer:
    def __init__(self):
        pass

    def clean_image_1(self, image):
        _, th1 = cv2.threshold(image, 17, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        th1 = cv2.dilate(th1, kernel, iterations=1)
        return th1

    def clean_image_2(self, image):
        hist = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_hist = clahe.apply(hist)
        # plot_image(clahe_hist)
        th1 = cv2.adaptiveThreshold(clahe_hist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        # plot_image(th1)
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
        # plot_image(closed_mask)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255), thickness=-1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def clean_image_3(self, image):
        img_path = f"in_img/hearts 3.png"
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        # plot_image(img[:, :, 1])
        _, th1 = cv2.threshold(img[:, :, 1], 100, 255, cv2.THRESH_BINARY_INV)
        return th1

    def clean_image_4(self, img):
        img_path = f"in_img/hearts 4.png"
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        sum_img = img[:, :, 1] - img[:, :, 2]
        plot_image(img[:, :, 1])
        plot_image(sum_img)
        _, tozero = cv2.threshold(sum_img, 160, 255, cv2.THRESH_TOZERO_INV)
        plot_image(tozero)
        _, th1 = cv2.threshold(tozero, 30, 255, cv2.THRESH_BINARY)
        return th1

    def clean_image_5(self, image):
        eq_img = cv2.equalizeHist(image)
        # kernel = np.ones((5, 5), np.uint8)
        # image = cv2.dilate(eq_img, kernel, iterations=1)
        return eq_img

    def clean_images(self, images):
        cleaned_img_1 = self.clean_image_1(images[0])
        plot_image(cleaned_img_1)
        cv2.imwrite(r'out_img/cleaned_hears_1.png', cleaned_img_1)
        cleaned_img_2 = self.clean_image_2(images[1])
        plot_image(cleaned_img_2)
        cv2.imwrite(r'out_img/cleaned_hears_2.png', cleaned_img_2)
        cleaned_img_3  = self.clean_image_3(images[2])
        plot_image(cleaned_img_3)
        cv2.imwrite(r'out_img/cleaned_hears_3.png', cleaned_img_3)
        cleaned_img_4 = self.clean_image_4(images[3])
        plot_image(cleaned_img_4)
        cv2.imwrite(r'out_img/cleaned_hears_4.png', cleaned_img_4)
        cleaned_img_5 = self.clean_image_5(images[4])
        plot_image(cleaned_img_5)
        cv2.imwrite(r'out_img/cleaned_hears_5.png', cleaned_img_5)

    def detect_hearts(self):
        kernel = np.ones((5, 5), np.uint8)

        image = cv2.imread(r'out_img/cleaned_hears_4.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        edged = cv2.dilate(gray, kernel, iterations=1)
        edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        colors = [
            (255, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 198, 0),
            (128, 0, 128), (0, 255, 128), (0, 255, 128)]

        contours = list(contours)
        masterpieces = [contours[0]]

        def is_new(new_item):
            decision = True
            for j in masterpieces:
                dist = cv2.matchShapes(j, new_item, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < 0.0265:
                    decision = False
                    break
            return decision

        def find_best_match(item):
            match = 1
            best_match_index = 0
            for i in enumerate(masterpieces):
                dist = cv2.matchShapes(i[1], item, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < match:
                    match = dist
                    best_match_index = i[0]
            return best_match_index

        color_index = 0
        for i in contours:
            cur_item = i
            if is_new(cur_item):
                masterpieces.append(cur_item)
                color_index += 1

        print(len(masterpieces))
        for j in contours:
            cv2.fillPoly(image, [j], colors[find_best_match(j)])
        plot_image(image)




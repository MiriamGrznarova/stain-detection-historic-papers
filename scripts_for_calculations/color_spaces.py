import cv2
import numpy as np
import os

input_folder = '../test_data_for_preprocessing'
output_folder = 'output_color_spaces'
os.makedirs(output_folder, exist_ok=True)

image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

methods = {
    'HSV': lambda img: cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array([0, 5, 135]),
                                   np.array([50, 35, 180])),
    'LAB': lambda img: cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), np.array([140, 125, 130]),
                                   np.array([165, 135, 135])),
    'YCrCb': lambda img: cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), np.array([125, 125, 120]),
                                     np.array([160, 135, 125]))
}

with open(os.path.join(output_folder, 'contour_counts.txt'), 'w') as result_file:
    for method_name, method in methods.items():
        result_file.write(f"{method_name}\n")
        for img_file in image_files:
            image = cv2.imread(img_file)

            mask = method(image)
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output_image = image.copy()

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            result_file.write(f"  {os.path.basename(img_file)}: {len(contours)}\n")

            contours_output_path = os.path.join(output_folder,
                                                f'{os.path.basename(img_file)}_{method_name}_contours.jpg')
            cv2.imwrite(contours_output_path, output_image)

        result_file.write("\n")


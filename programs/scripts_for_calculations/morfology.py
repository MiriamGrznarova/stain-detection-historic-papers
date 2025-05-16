import cv2
import numpy as np
import os

input_folder = '../test_data_for_preprocessing'
output_folder = 'output_morphology'
os.makedirs(output_folder, exist_ok=True)

image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

with open(os.path.join(output_folder, 'contour_counts.txt'), 'w') as result_file:
    for img_file in image_files:
        image = cv2.imread(img_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((4, 4), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=2)

        mask_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        mask_output_path = os.path.join(output_folder, f"{os.path.basename(img_file)}_mask.jpg")
        cv2.imwrite(mask_output_path, mask_bgr)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = image.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        contours_output_path = os.path.join(output_folder, f"{os.path.basename(img_file)}_contours.jpg")
        cv2.imwrite(contours_output_path, output)
        result_file.write(f"{os.path.basename(img_file)}: {len(contours)}\n")
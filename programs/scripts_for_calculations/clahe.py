import cv2
import os

input_folder = '../test_data_for_preprocessing'
output_folder = 'output_contrast'
os.makedirs(output_folder, exist_ok=True)

image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

with open(os.path.join(output_folder, 'contour_counts.txt'), 'w') as result_file:
    for img_file in image_files:
        image = cv2.imread(img_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
        enhanced_gray = clahe.apply(gray)

        _, thresh = cv2.threshold(enhanced_gray, 150, 255, cv2.THRESH_BINARY_INV)

        mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        mask_output_path = os.path.join(output_folder, f"{os.path.basename(img_file)}_mask.jpg")
        cv2.imwrite(mask_output_path, mask_bgr)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = image.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        contours_output_path = os.path.join(output_folder, f"{os.path.basename(img_file)}_contours.jpg")
        cv2.imwrite(contours_output_path, output)

        result_file.write(f"{os.path.basename(img_file)}: {len(contours)}\n")
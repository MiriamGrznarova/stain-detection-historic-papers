import cv2
import os

input_folder = '../test_data_for_preprocessing'
output_folder = 'output_adaptive'
os.makedirs(output_folder, exist_ok=True)

image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

methods = {
    'mean_c': lambda img: cv2.adaptiveThreshold(img, 125, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 7),
    'gaussian_c': lambda img: cv2.adaptiveThreshold(img, 125, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51,
                                                    7),
    'mean_c_blur': lambda img: cv2.adaptiveThreshold(cv2.medianBlur(img, 5), 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY_INV, 51, 7)
}

with open(os.path.join(output_folder, 'contour_counts.txt'), 'w') as result_file:
    for method_name, method in methods.items():
        result_file.write(f"{method_name}\n")
        for img_file in image_files:
            image = cv2.imread(img_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            thresh = method(gray)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_with_contours = image.copy()
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

            result_file.write(f"  {os.path.basename(img_file)}: {len(contours)}\n")

            contours_output_path = os.path.join(output_folder,
                                                f'{os.path.basename(img_file)}_{method_name}_contours.jpg')
            cv2.imwrite(contours_output_path, image_with_contours)

        result_file.write("\n")


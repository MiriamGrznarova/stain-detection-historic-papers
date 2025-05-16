import os
import cv2
import numpy as np


def process_images(input_folder, output_folder, class_label):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_hsv = np.array([0, 5, 135])
            upper_hsv = np.array([50, 255, 255])

            mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output_image = image.copy()

            txt_filename = os.path.join(output_folder, f"{filename.split('.')[0]}.txt")

            with open(txt_filename, 'w') as txt_file:
                for contour in contours:
                    if 100 < cv2.contourArea(contour) < 40000:
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > h:
                            factor = w / h
                        else:
                            factor = h / w
                        if factor < 3:
                            img_height, img_width = image.shape[:2]
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            width = w / img_width
                            height = h / img_height

                            txt_file.write(f"{class_label} {x_center} {y_center} {width} {height}\n")

                            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            contour_image_path = os.path.join(output_folder, f"{filename.split('.')[0]}_contour.jpg")
            cv2.imwrite(contour_image_path, output_image)
            original_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(original_image_path, image)


def main():
    dataset_folder = fr""
    output_folder = fr""
    class_label = 0  # foing


    input_folder = os.path.join(dataset_folder)
    output_subfolder = os.path.join(output_folder)

    if os.path.isdir(input_folder):
        process_images(input_folder, output_subfolder, class_label)

    # dataset_folder = 'sorted_dataset'
    # output_folder = 'annotated_dataset'
    # class_label = 0  # foxing
    #
    # for subdir in os.listdir(dataset_folder):
    #     if subdir in ['F_V', 'foxing', 'foxing_for_train']:
    #
    #         input_folder = os.path.join(dataset_folder, subdir)
    #         output_subfolder = os.path.join(output_folder, subdir)
    #
    #         if os.path.isdir(input_folder):
    #             process_images(input_folder, output_subfolder, class_label)


if __name__ == "__main__":
    main()

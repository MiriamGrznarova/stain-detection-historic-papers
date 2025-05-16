import os
import cv2

def normalize_image_size(image, size_target):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = size_target
        new_height = int(size_target / aspect_ratio)
    else:
        new_height = size_target
        new_width = int(size_target * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def process_images(input_fold, output_fold, size_target):
    if not os.path.exists(output_fold):
        os.makedirs(output_fold)

    for filename in os.listdir(input_fold):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_fold, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            normalized_image = normalize_image_size(image, size_target)
            output_path = os.path.join(output_fold, filename)
            cv2.imwrite(output_path, normalized_image)
            print(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    input_folder = r""
    output_folder = r""
    target_size = 1280

    process_images(input_folder, output_folder, target_size)

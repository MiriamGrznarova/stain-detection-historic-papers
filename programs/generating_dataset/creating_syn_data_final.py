import os
import random

import cv2
import numpy as np
from abc import ABC, abstractmethod


MAX_HEIGHT_OF_BG = 1000
MAX_SIZE_SAMPLES = 100


class Sample:
    def __init__(self, image_path, x, y, width, height, class_id):
        self.image_path = image_path
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id

    def get_image_path(self):
        return self.image_path

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_class_id(self):
        return self.class_id

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_class_id(self, class_id):
        self.class_id = class_id

    def __str__(self):
        return f"{self.image_path}, {self.x}, {self.y}, {self.width}, {self.height}, {self.class_id}"

class ImageLoader:
    def __init__(self):
        pass

    def load_images_from_folder(self, folder, class_id = None):
        """Loads all images from a folder and returns a dictionary with file numbers as keys."""
        images = {}
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(folder, filename)
                file_number = int(os.path.splitext(filename)[0])
                if class_id is None:
                    images[file_number] = image_path
                else:
                    images[file_number] = self.create_sample(image_path, 0, 0, class_id)
        return images

    @staticmethod
    def create_sample(image_path, x, y, class_id):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return Sample(image_path, x, y, width, height, class_id)

class SampleTransformer:
    @staticmethod
    def random_transform(sample, interval):
        """Applies random transformations (scaling, rotation) to the sample."""
        height, width = sample.shape[:2]

        # Resize if sample is too large
        if height > MAX_SIZE_SAMPLES or width > MAX_SIZE_SAMPLES:
            sample = cv2.resize(sample, (height // 4, width // 4), interpolation=cv2.INTER_AREA)
            height, width = sample.shape[:2]

        # Random scaling
        scale_factor = random.uniform(interval[0], interval[1])
        new_size = (max(int(width * scale_factor), 1), max(int(height * scale_factor), 1))
        resized_sample = cv2.resize(sample, new_size, interpolation=cv2.INTER_AREA)

        # Random rotation
        angle = random.uniform(0, 360)
        center = (resized_sample.shape[1] // 2, resized_sample.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_sample = cv2.warpAffine(
            resized_sample, rotation_matrix, (resized_sample.shape[1], resized_sample.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        return rotated_sample, new_size


class SpotBlender:
    def __init__(self, tin_t_mapping):
        self.tint_mapping = tin_t_mapping

    @staticmethod
    def count_non_black_pixels(roi, threshold=10):
        non_black_pixels = np.sum(np.all(roi > threshold, axis=-1))
        total_pixels = roi.shape[0] * roi.shape[1]
        return non_black_pixels, total_pixels

    def get_tint_for_class(self, class_id, roi):
        base_tint = self.tint_mapping.get(class_id, np.array([0, 0, 0], dtype=np.float32))
        non_black_pixels, total_pixels = self.count_non_black_pixels(roi)
        non_black_ratio = non_black_pixels / total_pixels if total_pixels > 0 else 0

        if non_black_ratio > 0.8:
            adjusted_tint = base_tint * 0.6
        elif non_black_ratio > 0.5:
            adjusted_tint = base_tint * 0.8
        else:
            adjusted_tint = base_tint * 1.3

        return adjusted_tint

    def add_spots_with_blending(self, background, sample, x_offset, y_offset, alpha=0.8, radius_scale_factor=0.9, class_id=0):
        sh, sw = sample.shape[:2]
        roi = background[y_offset:y_offset + sh, x_offset:x_offset + sw]
        if roi.shape[:2] != sample.shape[:2]:
            raise ValueError("ROI size and sample size do not match!")

        if sample.shape[2] == 4:
            alpha_channel = sample[:, :, 3] / 255.0
        else:
            gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_sample, 1, 255, cv2.THRESH_BINARY)
            alpha_channel = binary_mask / 255.0

        # Gaussian gradient blending
        center = (sw // 2, sh // 2)
        radius = max(center) * radius_scale_factor
        y, x = np.ogrid[:sh, :sw]
        distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian_gradient = np.exp(-((distance_from_center ** 2) / (2 * (radius / 3) ** 2)))

        alpha_gradient = np.clip(alpha_channel * gaussian_gradient, 0, 1)

        # Apply class-specific tint
        tint = self.get_tint_for_class(class_id, roi)
        sample_float = np.clip(sample.astype(np.float32) + tint, 0, 255)

        # Blending
        roi_float = roi.astype(np.float32)
        blended = (
                roi_float * (1 - alpha * alpha_gradient[:, :, np.newaxis]) +
                sample_float * (alpha * alpha_gradient[:, :, np.newaxis])
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        background[y_offset:y_offset + sh, x_offset:x_offset + sw] = blended

        return background


class SyntheticSampleCreator(ABC):
    def __init__(self, background_images, sample_images, output_folde, tin_t_mapping):
        self.background_images = background_images
        self.sample_images = sample_images
        self.output_folder = output_folde
        self.spot_blender = SpotBlender(tin_t_mapping)

    @abstractmethod
    def create_samples(self):
        """Abstract method for creating synthetic samples."""
        pass

    @staticmethod
    def generate_yolo_annotations(image_path, annotations):
        """Generates YOLO annotations for the given image and annotations."""
        image = cv2.imread(image_path)
        image_shape = image.shape
        yolo_annotations = []
        for (class_id, x, y, w, h) in annotations:
            x_center = (x + w / 2) / image_shape[1]
            y_center = (y + h / 2) / image_shape[0]
            width = w / image_shape[1]
            height = h / image_shape[0]
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        annotation_file = os.path.splitext(image_path)[0] + ".txt"
        with open(annotation_file, "w") as f:
            f.write("\n".join(yolo_annotations))

    @staticmethod
    def check_overlap(spots_positions, x_offset, y_offset, sw, sh):
        for (x, y, w, h) in spots_positions:
            if not (x_offset + sw < x or x_offset > x + w or y_offset + sh < y or y_offset > y + h):
                return True
        return False


if __name__ == "__main__":
    name_and_int_class = ('foxing', 0)
    folder_path = r'..\syntetic_data\normalized_images'
    folder_path_for_samples = fr'..\syntetic_data\segmentated_samples\{name_and_int_class[0]}'
    output_folder = fr'..\syntetic_data\synthetic_output1\{name_and_int_class[0]}'

    tint_mapping = {
        0: np.array([0, 51, 153], dtype=np.float32),
        1: np.array([0, 20, 0], dtype=np.float32),
        2: np.array([10, 100, 100], dtype=np.float32),
        3: np.array([0, 0, 20], dtype=np.float32),
    }
    image_loader = ImageLoader()
    image_bg = image_loader.load_images_from_folder(folder_path, None)
    samples = image_loader.load_images_from_folder(folder_path_for_samples, name_and_int_class[1])

    creator = SyntheticSampleCreator(image_bg, samples, output_folder, tint_mapping)
    creator.create_samples()
    print(f"Synthetic images and annotations saved to: {output_folder}")

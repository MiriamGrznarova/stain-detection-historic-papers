import os
import random
from itertools import combinations
import cv2
import numpy as np
from creating_syn_data_final import SyntheticSampleCreator, SampleTransformer, ImageLoader
BOUNDING_BOXES = {}
class MultiClassSampleCreator(SyntheticSampleCreator):

    def create_samples(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for file_number, bg_path in self.background_images.items():
            self.process_single_background(file_number, bg_path)

    @staticmethod
    def resize_bounding_box(x, y, w, h, scale_factor):
        new_w = w * scale_factor
        new_h = h * scale_factor
        new_x = x - (new_w - w) / 2
        new_y = y - (new_h - h) / 2

        return new_x, new_y, new_w, new_h

    def process_single_background(self, file_number, bg_path):
        background = cv2.imread(bg_path)
        if background is None:
            print(f"Error loading background image: {bg_path}")
            return
        if not (background.shape[0] > 1280 or background.shape[1] > 1280):
            background =normalize_size_and_save(bg_path, bg_path, 1280)
        bg_height, bg_width = background.shape[:2]
        num_classes = len(self.sample_images)
        samples_per_class = random.randint(1, 20 // num_classes)

        selected_samples = []
        for class_id, samples_dict in self.sample_images.items():
            class_samples = list(samples_dict.values())
            if len(class_samples) <= samples_per_class:
                selected_samples.extend(class_samples)
            else:
                selected_samples.extend(random.sample(class_samples, samples_per_class))
        spots_positions = []
        annotations = []
        interval  = calculate_ratio(bg_width, bg_height)
        for sample in selected_samples:
            sample_image = cv2.imread(sample.get_image_path())
            if sample.class_id == 5:
                sample_value_factor = [interval, 0.8, 1.5]
            else:
                if background.shape[0] > 2500 or background.shape[1] > 2500:
                    sample_value_factor = [(interval[0] + 2, interval[1] + 2), 0.5, 0.9]
                else:
                    sample_value_factor = [(interval[0]+ 1, interval[1] + 1), 0.5, 0.9]
            transformed_sample, (sw, sh) = SampleTransformer.random_transform(sample_image, sample_value_factor[0])

            if sw > bg_width or sh > bg_height:
                continue

            for _ in range(10):  # Find non-overlapping position
                x_offset = random.randint(0, bg_width - sw)
                y_offset = random.randint(0, bg_height - sh)
                if not self.check_overlap(spots_positions, x_offset, y_offset, sw, sh):
                    background = self.spot_blender.add_spots_with_blending(
                        background, transformed_sample, x_offset, y_offset, sample_value_factor[1], sample_value_factor[2],class_id=sample.get_class_id()
                    )
                    spots_positions.append((x_offset, y_offset, sw, sh))
                    if sample.get_class_id() != 5:
                        newy_x, new_y, new_w, new_h = self.resize_bounding_box(x_offset, y_offset, sw, sh, sample_value_factor[2])
                        annotations.append((sample.get_class_id(), newy_x, new_y, new_w, new_h))
                    else:
                        newy_x, new_y, new_w, new_h = self.resize_bounding_box(x_offset, y_offset, sw, sh,
                                                                               2.2)
                        annotations.append((sample.get_class_id(), newy_x, new_y, new_w, new_h))
                    break

        class_names = "_".join(map(str, self.sample_images.keys()))
        if str(file_number) in BOUNDING_BOXES:
            annotations.extend(BOUNDING_BOXES[str(file_number)])
            output_file = os.path.join(self.output_folder, f'{file_number}_{class_names}_4.png')
        else:
            output_file = os.path.join(self.output_folder, f'{file_number}_{class_names}.png')
        cv2.imwrite(output_file, background)
        self.generate_yolo_annotations(output_file, annotations)

    @staticmethod
    def generate_yolo_annotations(image_path, annotations):
        """Generates YOLO annotations for the given image and annotations."""
        image_name = image_path.split("\\")
        image = cv2.imread(image_path)
        image_shape = image.shape
        yolo_annotations = []

        for (class_id, x, y, w, h) in annotations:
            if class_id != 4:
                x_center = (x + w / 2) / image_shape[1]
                y_center = (y + h / 2) / image_shape[0]
                width = w / image_shape[1]
                height = h / image_shape[0]
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            else:
                yolo_annotations.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        annotation_file = os.path.splitext(image_path)[0] + ".txt"
        with open(annotation_file, "w") as f:
            f.write("\n".join(yolo_annotations))

def normalize_size_and_save(image_path: str, output_path: str, target_max: int = 1280):
    """
    Normalizuje obrázok tak, že ak je jeden rozmer menší ako target_max,
    nastaví väčší rozmer na target_max a druhý upraví podľa pomeru strán.
    Obrázok uloží na output_path.
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    if width > height:
        new_width = target_max
        new_height = int(target_max * (height / width))
    else:
        new_height = target_max
        new_width = int(target_max * (width / height))

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_image)

    new_image = cv2.imread(output_path)
    return new_image

def calculate_ratio(dc_width, dc_height, min_ratio=0.002, max_ratio=0.006, reference_size=1280, sample_size=20):
    maxi_dc = max(dc_width, dc_height)

    # Škálovací faktor, ktorý zabezpečí správne zmenšovanie/zväčšovanie vzorky
    scale_factor = (maxi_dc / reference_size) ** 0.5  # odmocnina z pomeru veľkostí

    # Intervaly upravené podľa veľkosti pozadia
    mini = (min_ratio * maxi_dc) / scale_factor
    maxi = (max_ratio * maxi_dc) / scale_factor
    return mini/sample_size, maxi/sample_size



if __name__ == "__main__":
    # with open("bounding_boxes_zateceniny.json", "r") as file:
    #     BOUNDING_BOXES = json.load(file)
    # for key in BOUNDING_BOXES:
    #     BOUNDING_BOXES[key] = [(4,) + tuple(box.values()) for box in BOUNDING_BOXES[key]]
    folder_path = r'large_newspaper'
    image_loader = ImageLoader()
    background_images = image_loader.load_images_from_folder(folder_path)
    folder_path_for_samples = [r'segmentated_samples\foxing',
                               r'segmentated_samples\huby',
                               r'segmentated_samples\bacterie',
                               r'segmentated_samples\plesne',
                               r'segmentated_samples\vykaly']
    sample_images = {
        0: image_loader.load_images_from_folder(folder_path_for_samples[0], class_id=0),
        1: image_loader.load_images_from_folder(folder_path_for_samples[1], class_id=1),
        2: image_loader.load_images_from_folder(folder_path_for_samples[2], class_id=2),
        3: image_loader.load_images_from_folder(folder_path_for_samples[3], class_id=3),
        5: image_loader.load_images_from_folder(folder_path_for_samples[4], class_id=5)
    }
    # classes = list(sample_images.keys())
    # all_class_comb = [comb for i in range(1, len(classes) + 1) for comb in combinations(classes, i)]
    all_class_comb = [(0,), (5,), (0,5)]
    tint_mapping = {
        0: np.array([0, 51, 153], dtype=np.float32),
        1: np.array([0, 20, 0], dtype=np.float32),
        2: np.array([10, 100, 100], dtype=np.float32),
        3: np.array([0, 0, 20], dtype=np.float32),
        5: np.array([0, 0, 9], dtype=np.float32)
    }

    for combinations in all_class_comb:
        output_folder  = fr'less_BB_syn_data\{"_".join(map(str, combinations))}'
        sample_images_comb = {key: sample_images[key] for key in combinations}
        generator = MultiClassSampleCreator(background_images, sample_images_comb, output_folder, tint_mapping)
        generator.create_samples()
        print(f"Synthetic samples created in {output_folder}")
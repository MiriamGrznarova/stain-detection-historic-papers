# The script reduces the number of annotations in datasets by selecting the largest objects from each class, maintaining a representative proportion of annotations.
import os
import random
from glob import glob

DATASET_DIR = fr""
SPLITS = ["train", "valid"]
OUTPUT_DIR = fr""

random.seed(42)

for split in SPLITS:
    label_path = os.path.join(DATASET_DIR, split, "labels")
    output_path = os.path.join(OUTPUT_DIR, split)

    os.makedirs(output_path, exist_ok=True)

    txt_files = glob(os.path.join(label_path, "*.txt"))

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        class_entries = {}
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            area = float(parts[3]) * float(parts[4])
            if class_id not in class_entries:
                class_entries[class_id] = []
            class_entries[class_id].append((line, area))
        selected_lines = []
        for class_id, entries in class_entries.items():
            entries.sort(key=lambda x: x[1], reverse=True)
            to_keep = max(1, int(len(entries) * 0.1))
            selected_lines.extend([entry[0] for entry in entries[:to_keep]])
        filename = os.path.basename(txt_file)
        out_file = os.path.join(output_path, filename)
        selected_lines = [line if line.endswith('\n') else line + '\n' for line in selected_lines]

        with open(out_file, "w") as f:
            f.writelines(selected_lines)

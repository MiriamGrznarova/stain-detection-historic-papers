from roboflow import Roboflow
import os

rf = Roboflow(api_key="")
project = rf.workspace().project("")
base_folder_path = ""
allowed_folders = {""}
resume_from = ""
resume = True

for folder_name in os.listdir(base_folder_path):
    folder_path = os.path.join(base_folder_path, folder_name)

    if os.path.isdir(folder_path) and folder_name in allowed_folders:
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                if not resume and filename != resume_from:
                    continue
                resume = True

                image_path = os.path.join(folder_path, filename)
                annotation_path = os.path.join(folder_path, filename.replace(".png", ".txt"))

                metadata = {"folder_name": folder_name}

                try:
                    if os.path.exists(annotation_path):
                        project.upload(image_path, annotation_path=annotation_path, metadata=metadata)
                        print(f"Nahraný obrázok {filename} s anotáciou z priečinka {folder_name}.")
                    else:
                        project.upload(image_path, metadata=metadata)
                        print(f"Nahraný obrázok {filename} bez anotácie z priečinka {folder_name}.")
                except AttributeError as e:
                    print(f"Chyba pri nahrávaní {filename}: {e}")

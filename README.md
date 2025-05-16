# 🧠 Stain Detection on Historical Documents – Bachelor Thesis Scripts

This repository contains the most important Python scripts developed for my bachelor thesis:  
**"Identification and Analysis of Stains on Historical Documents Using Convolutional Neural Networks"**.

The goal of this thesis was to develop a method for identifying and analyzing stains on historical documents using convolutional neural networks (CNNs).  
By applying advanced image recognition techniques, the work aims to create an automated system capable of detecting and classifying various types of stains.  
This system can contribute to improved conservation and restoration efforts.  
The proposed solution was tested and evaluated based on accuracy and effectiveness in analyzing document degradation.

This work was conducted within the **HIPSI research project**:  
*“A system for identifying stains on historical paper based on artificial intelligence and bioinformatics for conservation and restoration”*  
(APVV-23-0250, 2024–2027), in collaboration with the **Institute of Molecular Biology of the Slovak Academy of Sciences (ÚMB SAV)** and the **Academy of Fine Arts and Design in Bratislava (VŠVU)**.

---

## 📁 Repository Structure

- `cleaning_normalization/`  
  Scripts used for normalizing image sizes and standardizing page dimensions.

- `editing_dataset/`  
  Utility scripts for dataset editing, counting bounding boxes per class, image statistics, and set balancing.

- `experiments_for_generating_annotations_on_real_data/`  
  Scripts used to pre-annotate real images provided by VŠVU using the selected best-performing method.

- `generating_dataset/`  
  Two key scripts for generating the **synthetic dataset** by combining segmented stain masks with clean document backgrounds.  
  (⚠️ Note: folders with segmanted stains and clean documents are not included due to storage constraints.)

- `options_of_image_preprocessing/`  
  Early experiments with preprocessing techniques on real data (e.g., CLAHE, morfological transformations).

- `scripts_for_calculations/`  
  Scripts used to calculate performance metrics and compare preprocessing methods.  
  The most effective method identified here was later used in the annotation experiments above.

---

## 🛠 Tools & AI Assistance

During development, I used:
- **PyCharm Copilot** for code completion and suggestions
- **ChatGPT** for debugging and error resolution

---

## 📄 License

This repository is provided for academic and non-commercial use.


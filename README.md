# 🧠 Stain Detection on Historical Documents – Bachelor Thesis Scripts

This repository contains the source code developed for the automatic detection of biological degradation in historical documents using deep learning and synthetic data generation.

The repository includes implementations for:

- semi-automatic annotation of historical document degradation,
- synthetic dataset generation using alpha blending and the proposed CMYK-based blending approach,
- object detection experiments using the YOLO family of models,
- evaluation of preprocessing methods and detection performance.

The work was carried out within the **HIPSI research project**

> *A system for identifying stains on historical paper based on artificial intelligence and bioinformatics for conservation and restoration*  
> APVV-23-0250 (2024–2027)

in collaboration with

- Comenius University Bratislava,
- Institute of Molecular Biology, Slovak Academy of Sciences,
- Academy of Fine Arts and Design in Bratislava.

---

## 📁 Repository Structure

- `cleaning_normalization/`  
  Scripts used for normalizing image sizes and standardizing page dimensions.

- `editing_dataset/`  
  Utility scripts for dataset editing, counting bounding boxes per class, image statistics, and set balancing.

- `experiments_for_generating_annotations_on_real_data/`  
  Scripts used to pre-annotate real images provided by VŠVU using the selected best-performing method.

- `generating_dataset/`  
  Implementation of the synthetic dataset generation methods.

  - alpha_blending/

    Baseline synthetic data generation using conventional alpha blending.

    Files:

    - `creating_syn_data_final.py`
    - `MultiClassSampleCreator.py`

  - cmyk_blending/

    Improved synthetic data generation based on the proposed CMYK blending strategy.

    Files:

    - `feces_foxing_script.py`
      - generation of Dataset A (foxing and insect-induced degradation)

    - `generator_cmyk_bac_fun.py`
      - generation of Dataset B (fungal and bacterial degradation)

    The proposed pipeline includes:

    - random scaling and rotation,
    - CMYK-based color adaptation,
    - irregular mask generation,
    - edge feathering,
    - texture preservation,
    - class-specific blending,
    - automatic YOLO annotation generation.

- `notebooks/`
Training notebooks used for YOLO experiments with different model architectures and dataset configurations.

- `options_of_image_preprocessing/`  
  Early experiments with preprocessing techniques on real data (e.g., CLAHE, morfological transformations).

- `scripts_for_calculations/`  
  Scripts used to calculate performance metrics and compare preprocessing methods.  
  The most effective method identified here was later used in the annotation experiments above.

---
## Notes

Large datasets, segmented stain samples, trained models, and historical document collections are not included in this repository due to storage limitations and copyright restrictions.

## Related Publication

This repository accompanies the research presented in:

**Synthetic Data Generation for Biological Stain Detection in Historical Documents**

## 🛠 Tools & AI Assistance

During development, I used:
- **PyCharm Copilot** for code completion and suggestions
- **ChatGPT** for debugging and error resolution

---

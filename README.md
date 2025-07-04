## Goal
In this project, we aim to build a modular deep learning system for multi-task learning that can simultaneously perform disease grading (multi-class classification) and lesion segmentation (binary or multi-label) from retinal images. The system should be capable of handling both single-task learning and multi-task learning scenarios.

## Dataset
We utilize the Indian Diabetic Retinopathy Image Dataset (IDRiD), which can be downloaded [here](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid). Make sure the dataset follows this structure.
dataset/
├── Disease_Grading/
│ ├── ground_truth/
│ │ ├── disease_grading_train.csv
│ │ └── disease_grading_test.csv
│ └── original_images/
│ ├── Train/
│ └── Test/
├── Segmentation/
│ ├── ground_truth/
│ │ ├── Train/
│ │ │ ├── Haemorrhages/
│ │ │ ├── Hard_Exudates/
│ │ │ ├── Microaneurysms/
│ │ │ ├── Optic_Disc/
│ │ │ └── Soft_Exudates/
│ │ └── Test/
│ │ ├── Haemorrhages/
│ │ ├── Hard_Exudates/
│ │ ├── Microaneurysms/
│ │ ├── Optic_Disc/
│ │ └── Soft_Exudates/
│ └── original_images/
│ ├── Train/
│ └── Test/



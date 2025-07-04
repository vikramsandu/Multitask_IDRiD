# Imports
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class MultiTaskIDRiDDataset(Dataset):
    """
    Given image and ground truth paths for Disease Grading and
    Multi-Label Lesion Segmentation, this dataset class reads,
    preprocesses, and prepares the data for efficient Multi-Task
    training via a DataLoader.

    This class also allows us to selectively load the data for
    Single Task Training. (If corresponding image paths are provides
    as None Value)
    """

    def __init__(self,
                 disease_grading_image_paths=None,
                 disease_grading_ground_truth_csv_path=None,
                 disease_grading_transform=None,
                 lesion_segment_image_paths=None,
                 lesion_segment_ground_truth_path=None,
                 lesion_segment_transform=None
                 ):
        # Initialize Data for Disease Grading Task
        if disease_grading_image_paths is not None:
            self.disease_grading_image_paths = disease_grading_image_paths
            self.disease_grading_csv = pd.read_csv(disease_grading_ground_truth_csv_path)
            self.disease_grading_transform = disease_grading_transform

        # Initialize Data for Multi-Label Lesion Segmentation Task
        if lesion_segment_image_paths is not None:
            self.lesion_segment_image_paths = lesion_segment_image_paths
            self.lesion_segment_ground_truth_path = lesion_segment_ground_truth_path
            # Handle Multi-Label Lesion Segmentation (5 Classes)
            self.lesion_segment_classes = ["Haemorrhages", "Hard_Exudates", "Microaneurysms",
                                           "Optic_Disc", "Soft_Exudates"]
            self.lesion_segment_transform = lesion_segment_transform

    def __len__(self):
        # Should be able to handle Multi-Task Training.
        grading_len = len(os.listdir(self.disease_grading_image_paths)) if hasattr(self,
                                                                                   'disease_grading_image_paths') else 0
        segment_len = len(os.listdir(self.lesion_segment_image_paths)) if hasattr(self,
                                                                                  'lesion_segment_image_paths') else 0

        # Return max length to allow any combination of the two images
        return max(grading_len, segment_len)

    def __getitem__(self, idx):

        # Disease Grading Sample
        disease_grading_image, disease_grading_label = torch.tensor(0), torch.tensor(0)
        if hasattr(self, "disease_grading_image_paths"):
            # Make sure that index does not exceed the total number of images.
            disease_grading_idx = idx % len(os.listdir(self.disease_grading_image_paths))

            # Retrieve the Image metadata from the CSV.
            sample_metadata = self.disease_grading_csv.iloc[disease_grading_idx]
            image_name = sample_metadata["Image name"]
            retinopathy_grade = sample_metadata["Retinopathy grade"]
            # risk_of_macular_edema = sample_metadata["Risk of macular edema "]  # Space in the end

            # Load Image. (Make sure Images are in "jpg" format) # Hardcoded
            image_path = os.path.join(self.disease_grading_image_paths, f"{image_name}.jpg")
            disease_grading_image = Image.open(image_path).convert("RGB")

            # Transform if any. (Tensor + Resize + Normalize + Data Augmentations)
            if self.disease_grading_transform:
                disease_grading_image = self.disease_grading_transform(disease_grading_image)

            # Load labels for Multi-class classification.
            disease_grading_label = torch.tensor(retinopathy_grade, dtype=torch.long)

        # Lesion Segmentation Sample.
        lesion_segment_image, lesion_segment_mask = torch.tensor(0), torch.tensor(0)
        if hasattr(self, "lesion_segment_image_paths"):
            # Make sure that index does not exceed the total number of images.
            lesion_segment_idx = idx % len(os.listdir(self.lesion_segment_image_paths))

            # Load Image (or simple alternative random.sample could also be used.)
            image_name = os.listdir(self.lesion_segment_image_paths)[lesion_segment_idx]
            image_path = os.path.join(self.lesion_segment_image_paths, image_name)
            lesion_segment_image = Image.open(image_path).convert('RGB')
            img_size = lesion_segment_image.size

            # Load Multi-label segmentations Mask also.
            lesion_segment_mask_list = []
            mask_identifier = image_name.split('.')[0]
            for cls in self.lesion_segment_classes:
                # Select Corresponding mask of the image.
                masks_path = os.path.join(self.lesion_segment_ground_truth_path, cls)
                mask_name = [m for m in os.listdir(masks_path) if m.startswith(mask_identifier)]

                # Handle for empty masks in Soft Exudates
                if len(mask_name) != 0:
                    mask = Image.open(os.path.join(masks_path, mask_name[0])).convert('L')
                else:
                    mask = Image.new('L', img_size, 0)

                # PIL Mask List
                lesion_segment_mask_list.append(mask)

            # Joint Transform if any. (Must include for Data Augmentation)
            # Since Random Crop, Flip, etc. should be applied to the both image and mask.
            if self.lesion_segment_transform:
                # Output should be torch tensor.
                lesion_segment_image, lesion_segment_mask = self.lesion_segment_transform(lesion_segment_image,
                                                                                          lesion_segment_mask_list
                                                                                          )
        # Return
        return {"disease_grading_image": disease_grading_image,
                "disease_grading_label": disease_grading_label,
                "lesion_segment_image": lesion_segment_image,
                "lesion_segment_mask": lesion_segment_mask
                }

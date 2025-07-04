import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import torch


def get_classification_transforms(resize_dim, use_aug=False):
    """
    Augmentation for the Disease Grading Classification
    :param resize_dim:
    :param use_aug:
    :return:
    """
    if use_aug:
        # Train Augmentation
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(resize_dim, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    return preprocess


class SegmentationJointTransform:
    def __init__(self,
                 resize=None,
                 crop_size=None,
                 normalize="imagenet",
                 use_aug=True  # Flag to control augmentations
                 ):
        self.resize = resize
        self.crop_size = crop_size
        self.use_aug = use_aug

        # Define normalization
        if normalize == "imagenet":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        else:
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])

    def __call__(self, image: Image, mask_pil_list: list):
        """
        image: PIL RGB image
        mask_pil_list: list of PIL 'L' mode masks (1 per class)
        """

        # Convert masks to tensor [1, H, W] and stack to [C, H, W]
        mask_tensor_list = [transforms.ToTensor()(mask) for mask in mask_pil_list]
        mask_tensor = torch.cat(mask_tensor_list, dim=0)

        # Resize image and mask
        if self.resize is not None:
            image = TF.resize(image, (self.resize, self.resize))
            mask_tensor = TF.resize(mask_tensor, (self.resize, self.resize),
                                    interpolation=TF.InterpolationMode.NEAREST)

        # Apply augmentations only in training mode
        if self.use_aug:
            # Random crop
            if self.crop_size is not None:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                mask_tensor = TF.crop(mask_tensor, i, j, h, w)

            # Horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask_tensor = TF.hflip(mask_tensor)

            # Random rotation (±15 degrees)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)

            # Step 5: Blur
            if random.random() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.choice([1, 2])))

            # Step 6: Brightness & Contrast
            if random.random() < 0.25:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.9, 1.1))

                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.9, 1.1))

        # To Tensor and Normalize image
        image = TF.to_tensor(image)
        image = self.normalize(image)

        # Binarize masks
        mask_tensor = (mask_tensor > 0).float()

        return image, mask_tensor

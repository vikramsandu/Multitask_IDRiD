# Import utilities
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import os
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils.dataset import MultiTaskIDRiDDataset
from utils.data_augmentation import SegmentationJointTransform, get_classification_transforms
from utils.model import MultiTaskIDRiDModel
from utils.metrics import compute_dice

# For debugging.
from matplotlib import pyplot as plt


# Setup random seed for reproducibility.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Training Loop.
def train(config_path: Path):

    # Load configs.
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Create output directory and save configs. (for reproducibility)
    exp_dirpath = config.get("exp_path")
    os.makedirs(exp_dirpath, exist_ok=True)
    with open(os.path.join(exp_dirpath, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Check Device and set to cuda if available.
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Preprocessing utilities.
    # Disease Grading: Resize + Tensor + Normalize.
    use_aug = config["dataset"]["augmentation"]
    resize_dim = config["dataset"]["resize_classification"]

    # Augmentations for Disease Grading classification
    classification_transform_train = get_classification_transforms(resize_dim, use_aug=use_aug)
    classification_transform_test = get_classification_transforms(resize_dim, use_aug=False)

    # Augmentations for Lesion Segmentation.
    segmentation_transform_train = SegmentationJointTransform(
        resize=config["dataset"]["resize_segmentation"],
        crop_size=config["dataset"]["random_crop_segmentation"],
        normalize=config["dataset"]["normalize"],
        use_aug=use_aug
    )

    segmentation_transform_test = SegmentationJointTransform(
        resize=config["dataset"]["resize_segmentation"],
        crop_size=config["dataset"]["random_crop_segmentation"],
        normalize=config["dataset"]["normalize"],
        use_aug=False
    )

    # Initialize Dataset class
    # Train
    train_dataset = MultiTaskIDRiDDataset(
        disease_grading_image_paths=config["dataset"]["train_disease_img"],
        disease_grading_ground_truth_csv_path=config["dataset"]["train_disease_csv"],
        disease_grading_transform=classification_transform_train,
        lesion_segment_image_paths=config["dataset"]["train_seg_img"],
        lesion_segment_ground_truth_path=config["dataset"]["train_seg_mask"],
        lesion_segment_transform=segmentation_transform_train
    )

    # Validation
    val_dataset = MultiTaskIDRiDDataset(
        disease_grading_image_paths=config["dataset"]["test_disease_img"],
        disease_grading_ground_truth_csv_path=config["dataset"]["test_disease_csv"],
        disease_grading_transform=classification_transform_test,
        lesion_segment_image_paths=config["dataset"]["test_seg_img"],
        lesion_segment_ground_truth_path=config["dataset"]["test_seg_mask"],
        lesion_segment_transform=segmentation_transform_test
    )

    # Dataloader
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the Model.
    model = MultiTaskIDRiDModel(
        encoder_name=config["model"]["encoder_name"],
        num_class_labels=config["model"]["num_class_labels"],
        num_segmentation_classes=config["model"]["num_segmentation_classes"],
        pretrained=config["model"]["pretrained"],
        isClassify=config["model"]["isClassify"],
        isSegment=config["model"]["isSegment"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Criterion.
    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = nn.BCEWithLogitsLoss()

    # Optimizer + Schedular
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Early stopping
    patience = config["training"]["early_stopping_patience"]
    early_stop_counter = 0

    # Log everything for further analysis.
    # Logs dictionary
    logs = {
        "epoch": [],
        "train_classification_loss": [],
        "train_segmentation_loss": [],
        "train_accuracy": [],
        "train_dice": [],
        "train_total_loss": [],
        "val_classification_loss": [],
        "val_segmentation_loss": [],
        "val_accuracy": [],
        "val_dice": [],
        "val_total_loss": []
    }

    # Training Loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    lambda_class = config["training"]["lambda_class"]

    for epoch in range(config["training"]["num_epochs"]):
        # Set Model to train mode.
        model.train()

        # Logging helpers.
        cls_losses, seg_losses = [], []
        y_true_cls, y_pred_cls = [], []
        seg_dices = []

        # Loop through the batches.
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Train"):

            # Clear up accumulates gradients from prev steps if any.
            optimizer.zero_grad()

            # Read batches and move to device.
            cls_img = batch["disease_grading_image"].to(device)
            cls_label = batch["disease_grading_label"].to(device)
            seg_img = batch["lesion_segment_image"].to(device)
            seg_mask = batch["lesion_segment_mask"].to(device)

            # Predictions.
            outputs = model(disease_grading_input=cls_img, lesion_segment_input=seg_img)

            # Post-processing + Calculate loss
            total_loss = 0.0
            if "disease_grading_output" in outputs:
                cls_out = outputs["disease_grading_output"]
                cls_loss = classification_criterion(cls_out, cls_label)
                cls_losses.append(cls_loss.item())
                total_loss += lambda_class * cls_loss

                # Append
                y_true_cls.extend(cls_label.cpu().numpy())
                y_pred_cls.extend(torch.argmax(cls_out, dim=1).cpu().numpy())

            if "lesion_segmentation_output" in outputs:
                seg_out = outputs["lesion_segmentation_output"]
                seg_loss = segmentation_criterion(seg_out, seg_mask)
                seg_losses.append(seg_loss.item())
                total_loss += seg_loss

                # Append
                seg_dices.append(compute_dice(torch.sigmoid(seg_out), seg_mask))

            # Backprop + Update.
            total_loss.backward()
            optimizer.step()

        # Update schedular after every epoch.
        scheduler.step()

        # Logging - Train
        avg_cls_loss = sum(cls_losses) / len(cls_losses) if cls_losses else 0
        avg_seg_loss = sum(seg_losses) / len(seg_losses) if seg_losses else 0
        avg_dice = sum(seg_dices) / len(seg_dices) if seg_dices else 0
        acc = accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0
        total_train_loss = avg_cls_loss * lambda_class + avg_seg_loss

        print(
            f"\nEpoch [{epoch + 1}] "
            f"Train  Classification Loss: {avg_cls_loss:.4f} | "
            f"Segmentation Loss: {avg_seg_loss:.4f} "
            f"| Accuracy: {acc:.4f} "
            f"| Dice Score: {avg_dice:.4f} "
            f"|Total Loss: {total_train_loss:.4f}"
        )

        # Validation Loop.
        # Set model to eval mode.
        model.eval()
        # Logging helpers
        cls_losses, seg_losses = [], []
        y_true_cls, y_pred_cls = [], []
        seg_dices = []

        with torch.no_grad():
            for batch in val_loader:
                cls_img = batch["disease_grading_image"].to(device)
                cls_label = batch["disease_grading_label"].to(device)
                seg_img = batch["lesion_segment_image"].to(device)
                seg_mask = batch["lesion_segment_mask"].to(device)

                outputs = model(disease_grading_input=cls_img, lesion_segment_input=seg_img)

                if "disease_grading_output" in outputs:
                    cls_out = outputs["disease_grading_output"]
                    cls_loss = classification_criterion(cls_out, cls_label)
                    cls_losses.append(cls_loss.item())

                    y_true_cls.extend(cls_label.cpu().numpy())
                    y_pred_cls.extend(torch.argmax(cls_out, dim=1).cpu().numpy())

                if "lesion_segmentation_output" in outputs:
                    seg_out = outputs["lesion_segmentation_output"]
                    seg_loss = segmentation_criterion(seg_out, seg_mask)
                    seg_losses.append(seg_loss.item())

                    seg_dices.append(compute_dice(torch.sigmoid(seg_out), seg_mask))

        val_cls_loss = sum(cls_losses) / len(cls_losses) if cls_losses else 0
        val_seg_loss = sum(seg_losses) / len(seg_losses) if seg_losses else 0
        val_dice = sum(seg_dices) / len(seg_dices) if seg_dices else 0
        val_acc = accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0
        total_val_loss = val_cls_loss * lambda_class + val_seg_loss

        print(
            f"Epoch [{epoch + 1}] "
            f"Val    Classification Loss: {val_cls_loss:.4f} | "
            f"Segmentation Loss: {val_seg_loss:.4f} | "
            f"Accuracy: {val_acc:.4f} | "
            f"Dice Score: {val_dice:.4f} | "
            f"Total Loss: {total_val_loss:.4f}"
        )

        # Store logs.
        logs["epoch"].append(epoch + 1)
        logs["train_classification_loss"].append(avg_cls_loss)
        logs["train_segmentation_loss"].append(avg_seg_loss)
        logs["train_accuracy"].append(acc)
        logs["train_dice"].append(avg_dice)
        logs["train_total_loss"].append(total_train_loss)
        logs["val_classification_loss"].append(val_cls_loss)
        logs["val_segmentation_loss"].append(val_seg_loss)
        logs["val_accuracy"].append(val_acc)
        logs["val_dice"].append(val_dice)
        logs["val_total_loss"].append(total_val_loss)

        # Dump logs in json file.
        with open(os.path.join(exp_dirpath, "logs.json"), "w") as f:
            json.dump(logs, f, indent=4)

        # Save best model based on total validation loss
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dirpath, f"best.pth"))
            print(f"✔️ Saved best total loss model at epoch {epoch + 1}: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1} after {patience} no-improve epochs.")
                break

        # Save best model based on classification accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(exp_dirpath, f"best_acc.pth"))
            print(f"✔️ Saved best accuracy model at epoch {epoch + 1}: {best_val_acc:.4f}")


if __name__ == "__main__":
    # Empty cache if any.
    torch.cuda.empty_cache()

    # Setup seed.
    setup_seed(42)

    # Provide configs path as input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)

    # Get configs Path.
    args = parser.parse_args()
    config_path = Path(args.config_path)

    # Train
    train(config_path)

    # All done
    print("\nTraining complete.")

import torch
import json
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from utils.dataset import MultiTaskIDRiDDataset
from utils.data_augmentation import SegmentationJointTransform, get_classification_transforms
from utils.model import MultiTaskIDRiDModel
from utils.metrics import compute_dice
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def validate(config_path: Path):

    with open(config_path) as f:
        config = json.load(f)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    classification_transform = get_classification_transforms(
        config["dataset"]["resize_classification"], use_aug=False
    )
    segmentation_transform = SegmentationJointTransform(
        resize=config["dataset"]["resize_segmentation"],
        crop_size=config["dataset"]["random_crop_segmentation"],
        normalize=config["dataset"]["normalize"],
        use_aug=False
    )

    val_dataset = MultiTaskIDRiDDataset(
        disease_grading_image_paths=config["dataset"]["test_disease_img"],
        disease_grading_ground_truth_csv_path=config["dataset"]["test_disease_csv"],
        disease_grading_transform=classification_transform,
        lesion_segment_image_paths=config["dataset"]["test_seg_img"],
        lesion_segment_ground_truth_path=config["dataset"]["test_seg_mask"],
        lesion_segment_transform=segmentation_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = MultiTaskIDRiDModel(
        encoder_name=config["model"]["encoder_name"],
        num_class_labels=config["model"]["num_class_labels"],
        num_segmentation_classes=config["model"]["num_segmentation_classes"],
        pretrained=config["model"]["pretrained"],
        isClassify=config["model"]["isClassify"],
        isSegment=config["model"]["isSegment"],
        dropout=config["model"]["dropout"]
    ).to(device)

    model_path = os.path.join(config["exp_path"], "best_acc.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(config["exp_path"], "best.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    classification_criterion = torch.nn.CrossEntropyLoss()
    segmentation_criterion = torch.nn.BCEWithLogitsLoss()
    lambda_class = config["training"]["lambda_class"]

    os.makedirs(os.path.join(config["exp_path"], "original_image"), exist_ok=True)
    os.makedirs(os.path.join(config["exp_path"], "gt_mask"), exist_ok=True)
    os.makedirs(os.path.join(config["exp_path"], "pred_mask"), exist_ok=True)

    cls_losses, seg_losses = [], []
    y_true_cls, y_pred_cls = [], []
    seg_dices = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Validation")):

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

                # Save image, GT and prediction
                vutils.save_image(batch["lesion_segment_image"], os.path.join(config["exp_path"], "original_image", f"{idx:04d}.png"))
                vutils.save_image(torch.cat([batch["lesion_segment_mask"][:, i:i+1] for i in range(batch["lesion_segment_mask"].size(1))], dim=-1),
                                  os.path.join(config["exp_path"], "gt_mask", f"{idx:04d}.png"))
                binarized = [(torch.sigmoid(seg_out[:, i:i + 1]) > 0.5).float() for i in range(seg_out.size(1))]
                vutils.save_image(torch.cat(binarized, dim=-1),
                                  os.path.join(config["exp_path"], "pred_mask", f"{idx:04d}.png"))

    val_cls_loss = sum(cls_losses) / len(cls_losses) if cls_losses else 0
    val_seg_loss = sum(seg_losses) / len(seg_losses) if seg_losses else 0
    val_dice = sum(seg_dices) / len(seg_dices) if seg_dices else 0
    val_acc = accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0
    total_val_loss = val_cls_loss * lambda_class + val_seg_loss

    print(f"\n\u2705 Validation Results:")
    print(f"Classification Loss : {val_cls_loss:.4f}")
    print(f"Segmentation Loss   : {val_seg_loss:.4f}")
    print(f"Accuracy            : {val_acc:.4f}")
    print(f"Dice Score          : {val_dice:.4f}")
    print(f"Total Loss          : {total_val_loss:.4f}")

    metrics = {
        "val_classification_loss": val_cls_loss,
        "val_segmentation_loss": val_seg_loss,
        "val_accuracy": val_acc,
        "val_dice": val_dice,
        "val_total_loss": total_val_loss
    }

    with open(os.path.join(config["exp_path"], "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(config["exp_path"], "confusion_matrix.png"))
    print(f"\n Saved validation metrics and confusion matrix to {config['exp_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to config.json")
    args = parser.parse_args()

    validate(Path(args.config_path))

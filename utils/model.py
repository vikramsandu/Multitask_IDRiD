# Imports
import torch
import torch.nn as nn
# Make sure to install this using "pip install segmentation-models-pytorch"
import segmentation_models_pytorch as smp
import torch.nn.functional as F


class MultiTaskIDRiDModel(nn.Module):
    """
    Design:
    1) Shared Backbone for both tasks.
    2) Dynamic Routing.
    3) Two experts i) Disease Grading and ii) Segmentation
    4) Should be able to support Single Task also.
    """

    def __init__(self,
                 encoder_name,  # Shared Backbone
                 num_class_labels=5,  # Disease Grading Classes (Retinopathy Grade)
                 num_segmentation_classes=5,  # Lesion Segmentation Classes
                 pretrained=None,  # Either to train from the scratch or use pretrained weights such as imagenet.
                 isClassify=True,  # Supports classification
                 isSegment=True,  # Supports segmentation
                 dropout=0.0  # Use Dropout for classification head
                 ):
        super(MultiTaskIDRiDModel, self).__init__()

        # Load Model using smp library
        base = smp.Unet(encoder_name, encoder_weights=pretrained, in_channels=3,
                        classes=num_segmentation_classes if isSegment else 0)

        # Extract Encoder + Decoder + Segmentation head
        self.encoder = base.encoder
        self.decoder = base.decoder if isSegment else None
        self.seg_head = base.segmentation_head if isSegment else None

        # Classification head
        self.class_head = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(self.encoder.out_channels[-1], num_class_labels)
            ) if isClassify else None
        )

    def forward(self,
                disease_grading_input=None,
                lesion_segment_input=None
                ):

        outputs = {}

        # Feature extraction
        feats_cls = self.encoder(disease_grading_input)[-1] if self.class_head else None
        feats_seg_list = self.encoder(lesion_segment_input) if self.seg_head else None

        # Disease Grading classification output
        if self.class_head:
            outputs["disease_grading_output"] = self.class_head(feats_cls)

        # Lesion segmentation output
        if self.seg_head:
            outputs["lesion_segmentation_output"] = self.seg_head(self.decoder(feats_seg_list))

        return outputs

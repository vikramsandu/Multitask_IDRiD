# Run this bash file to produce results shown in the report for all experiments.

## Training ##
# Exp-1: Disease Grading classification.
#python train.py  --config-path  configs/classification_only.json

# Exp-2: Lesion Segmentation
#python train.py --config-path  configs/segmentation_only.json
#
## Exp-3: Multitask with loss_class + loss_seg.
#python train.py --config-path  configs/multitask_lambda_1_0.json
#python train.py --config-path  configs/multitask_lambda_1_0_no_aug.json
#
## Exp-4: Multitask with 0.1 x loss_class + loss_seg.
#python train.py --config-path  configs/multitask_lambda_0_1.json
#python train.py --config-path  configs/multitask_lambda_0_1_no_aug.json
#
## Exp-5: Multitask with 10 x loss_class + loss_seg.
#python train.py --config-path  configs/multitask_lambda_10_0.json

## Validation ##
#python validation.py  --config-path  configs/classification_only.json
#python validation.py  --config-path  configs/segmentation_only.json
#python validation.py  --config-path  configs/multitask_lambda_1_0.json
#python validation.py  --config-path  configs/multitask_lambda_1_0_no_aug.json
python validation.py  --config-path  configs/multitask_lambda_0_1.json
#python validation.py  --config-path  configs/multitask_lambda_0_1_no_aug.json
#python validation.py  --config-path  configs/multitask_lambda_10_0.json
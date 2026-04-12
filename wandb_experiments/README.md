# W&B experiment scripts (Assignment 2)

All scripts live in `wandb_experiments/` and are meant to generate the **interactive** plots/tables/images you’ll embed in your public W&B report.

Run scripts from the repo root, e.g.:
```powershell
python wandb_experiments/exp_2_1_batchnorm_activations.py --help
```
If you’re using the provided venv on Windows, prefer:
```powershell
.\.venv\Scripts\python wandb_experiments/exp_2_1_batchnorm_activations.py --help
```

## 2.1 BatchNorm: activations + max stable LR

Train curves + activation distributions (same fixed val image each epoch):
```powershell
python wandb_experiments/exp_2_1_batchnorm_activations.py --batchnorm on
python wandb_experiments/exp_2_1_batchnorm_activations.py --batchnorm off
```

LR range test (proxy for “maximum stable LR”):
```powershell
python wandb_experiments/exp_2_1_lr_range_test.py --batchnorm on
python wandb_experiments/exp_2_1_lr_range_test.py --batchnorm off
```

## 2.2 Dropout internal dynamics (p=0.0 / 0.2 / 0.5)

Run three separate runs (W&B compare overlays them automatically):
```powershell
python wandb_experiments/exp_2_2_dropout_dynamics.py --dropout_p 0.0
python wandb_experiments/exp_2_2_dropout_dynamics.py --dropout_p 0.2
python wandb_experiments/exp_2_2_dropout_dynamics.py --dropout_p 0.5
```

## 2.3 Transfer learning showdown (segmentation)

Strict feature extractor / partial fine-tune / full fine-tune:
```powershell
python wandb_experiments/exp_2_3_transfer_learning_segmentation.py --strategy strict
python wandb_experiments/exp_2_3_transfer_learning_segmentation.py --strategy partial --unfreeze_last_blocks 2
python wandb_experiments/exp_2_3_transfer_learning_segmentation.py --strategy full
```

## 2.4 Feature maps (first vs last conv)

Provide an image path (pick a dog image from the dataset or your own):
```powershell
python wandb_experiments/exp_2_4_feature_maps.py --ckpt checkpoints/classifier.pth --image_path data/oxford-iiit-pet/images/<dog>.jpg
```

## 2.5 Localization “detection” table (confidence + IoU)

Logs a W&B table with image overlays (GT=green, pred=red), IoU, and an MC-dropout confidence score:
```powershell
python wandb_experiments/exp_2_5_detection_table.py --ckpt checkpoints/localizer.pth --split test --num_images 10
```

## 2.6 Segmentation: Dice vs pixel accuracy + 5 samples

Option A (recommended): log samples during training (same script as 2.3):
```powershell
python wandb_experiments/exp_2_3_transfer_learning_segmentation.py --strategy strict --log_samples --num_samples 5
```

Option B: evaluation-only logging from an existing checkpoint:
```powershell
python wandb_experiments/exp_2_6_segmentation_eval.py --ckpt checkpoints/unet.pth --num_samples 5
```

## 2.7 Final pipeline showcase (3 novel images)

Use local paths or URLs (you can pass more than 3):
```powershell
python wandb_experiments/exp_2_7_pipeline_showcase.py --images path1.jpg path2.jpg path3.jpg
```

## 2.8 Meta-analysis helper (optional)

Pulls prior runs from your W&B project and logs a `runs_summary` table back into the project:
```powershell
python wandb_experiments/exp_2_8_meta_analysis.py --entity <your_entity> --project <your_project>
```

## Notes
- Use `--project`, `--entity`, `--name`, `--tags`, `--mode` on any script to control W&B logging.
- If you want faster iteration, use `--train_subset` / `--val_subset` on training scripts.

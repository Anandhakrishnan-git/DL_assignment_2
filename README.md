# DA6401 Assignment-2 Skeleton Guide

This repository is an instructional skeleton for building the complete visual perception pipeline on Oxford-IIIT Pet.


### ADDITIONAL INSTRUCTIONS FOR ASSIGNMENT2:
- Ensure VGG11 is implemented according to the official paper(https://arxiv.org/abs/1409.1556). The only difference being injecting BatchNorm and CustomDropout layers is your design choice.
- Train all the networks on normalized images as input (as the test set given by autograder will be normalized images).
- The output of Localization model = [x_center, y_center, width, height] all these numbers are with respect to image coordinates, in pixel space (not normalized)
- Train the object localization network with the following loss function: MSE + custom_IOU_loss.
- Make sure the custom_IOU loss is in range: [0,1]
- In the custom IOU loss, you have to implement all the two reduction types: ["mean", "sum"] and the default reduction type should be "mean". You may include any other reduction type as well, which will help your network learn better.
- multitask.py shd load the saved checkpoints (classifier.pth, localizer.pth, unet.pth), initialize the shared backbone and heads with these trained weights and do prediction.
- Keep paths as relative paths for loading in multitask.py
- Assume input image size is fixed according to vgg11 paper(can be hardcoded need not pass as args)
- Stick to the arguments of the functions and classes given in the github repo, if you include any additional arguments make sure they always have some default value.
- Do not import any other python packages apart from the ones mentioned in assignment pdf, if you do so the autograder will instantly crash and your submission will not be evaluated.
- The following classes will be used by autograder: 
    ```
        from models.vgg11 import VGG11
        from models.layers import CustomDropout
        from losses.iou_loss import IoULoss
        from multitask import MultiTaskPerceptionModel
    ```
- The submission link for this assignment will be available by Saturday(04/04/2026) on gradescope





### GENERAL INSTRUCTIONS:
- From this assignment onwards, if we find any wandb report which is private/inaccessible while grading, there wont be any second chance, that submission will be marked 0 for wandb marks.
- The entireity of plots presented in the wandb report should be interactive and logged in the wandb project. Any screenshot or images of plots will straightly be marked 0 for that question.
- Gradescope offers an option to activate whichever submission you want to, and that submission will be used for evaluation. Under any circumstances, no requests to be raised to TAs to activate any of your prior submissions. It is the student's responsibility to do so(if required) before submission deadline.
- Assignment2 discussion forum has been opened on moodle for any doubt clarification/discussion.   



# Assignment 2 – Submission Guidelines

Follow the steps below carefully:

---

## Step 1 – Google Drive Setup

Create a new folder in your Google Drive and upload all 3 model checkpoints to it:

- `classifier.pth`
- `localizer.pth`
- `unet.pth`

---

## Step 2 – Get Drive IDs

For each `.pth` file:

1. Click the three dots (**More actions**) next to the file
2. Click **Share → Share**
3. Set access to **Anyone with the link**
4. Copy the link and extract the ID

**Example:**
Link → https://drive.google.com/file/d/1t2EgeJ3TaYFSBQoC9o0ojd8Nn52XzV0i/view?usp=sharing
ID   → 1t2EgeJ3TaYFSBQoC9o0ojd8Nn52XzV0i

---

## Step 3 – Update Your Code

Paste these 4 lines at the **start** of the `init()` function inside `MultiTaskPerceptionModel`:
```python
import gdown
gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)
```

Replace each `<...drive id>` with the actual IDs from Step 2.

---

## Step 4 – Clean Up Locally

Delete all 3 `.pth` files from your local `/checkpoints` folder.

---

## Step 5 – Push to GitHub

Push the current project to GitHub. Make sure **no `.pth` files** are included.

---

## Step 6 – Verify Project Structure

Your final project should look like this:

```
.
├── checkpoints
│   └── checkpoints.md
├── data
│   └── pets_dataset.py
├── inference.py
├── losses
│   ├── __init__.py
│   └── iou_loss.py
├── models
│   ├── classification.py
│   ├── __init__.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── README.md
├── requirements.txt
└── train.py
```
---

## Step 7 – README

Make sure your README includes:

- Public **WandB report** link
- **GitHub repo** link

---

## Step 8 – Submit

Zip the project and submit on **Gradescope**.

> ⚠️ **Do NOT delete** the above created Drive folder till Assignment 2 marks are released.


# Contact

For questions or issues, please contact the teaching staff or post on the course forum.

---

Good luck with your implementation!
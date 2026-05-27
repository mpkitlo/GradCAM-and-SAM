# Grad-CAM + SAM: Image Segmentation Without User Input

An automated segmentation pipeline that combines Grad-CAM attention maps with Meta's Segment Anything Model (SAM) to segment objects without any manual prompts.

## Idea

Instead of requiring a user to click on the object to segment, this pipeline uses Grad-CAM heatmaps from a classifier to automatically generate point prompts for SAM. The Grad-CAM activation map highlights where the model "looks," and that information is converted into foreground/background points that SAM uses to produce a segmentation mask.

## Pipelines

Two approaches were implemented and compared:

- **FSamPipeline (Foreground-only):** The Grad-CAM map is normalized to [0,1] and binarized with a threshold of 0.6 (inspired by the weakly-supervised localization technique from the original Grad-CAM paper). The geometric centroid of the active region becomes the foreground prompt for SAM.

- **FBSamPipeline (Foreground + Background):** Extends the first pipeline by adding a negative (background) prompt — the image corner furthest from the foreground centroid — to help SAM suppress background noise.

## Results

| Metric | FSamPipeline | FBSamPipeline |
|---|---|---|
| Avg Hit Rate | 0.768 | 0.768 |
| Avg Distance | 5.269 | 5.269 |
| Avg IoU | **0.802** | 0.795 |

The foreground-only pipeline slightly outperformed the foreground+background variant (IoU 0.802 vs 0.795), suggesting that a naive corner-based background point can actually hurt segmentation quality.

## Tech Stack

Python, PyTorch, Grad-CAM, Segment Anything Model (SAM)

## Files

- `gradcam_sam.ipynb` — Full pipeline implementation and evaluation
- `Report.pdf` — [Detailed analysis and discussion](./Report.pdf)

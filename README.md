🔬 Skin Lesion Classification — ISIC 2020
<div align="center">
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=flat&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue?style=flat)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)
Pattern Recognition — Phase 2: Proposal and Code Implementation
Field	Details
Student	Nisarga Bhaktharahalli Muniraju
Programme	M.Sc. Data Science
University	University of Europe for Applied Sciences
Supervisor	Raja Hashim Ali
Date	June 2026
</div>
---
📋 Table of Contents
Project Overview
Dataset
Models
Results
Repository Structure
How to Run
Figures and Outputs
Dependencies
Disclaimer
---
🧠 Project Overview
This project builds a binary skin lesion classification system — distinguishing Benign from Malignant dermoscopic lesions — using four deep learning architectures evaluated on the ISIC 2020 dataset.
Key Features
✅ Binary classification: Benign vs. Malignant
✅ Four architectures compared: Custom CNN, MobileNetV2, EfficientNetB0, ResNet50
✅ Severe class imbalance handled via calibrated class weights (~98% benign / ~2% malignant)
✅ Dermoscopy-specific augmentation (360° rotation, colour jitter, channel shift)
✅ Two-phase transfer learning strategy (head warm-up → backbone fine-tuning)
✅ Grad-CAM visual explanations for all four models
✅ Full error analysis with false-positive / false-negative breakdown
⚠️ Decision-support tool only — not a clinical diagnostic replacement
---
📦 Dataset
Attribute	Details
Name	ISIC 2020 Skin Cancer Classification Dataset
Source	Kaggle
Total Images	33,126 dermoscopic images
Classes	2 — Benign (0), Malignant (1)
Image Size	256 × 256 px, JPEG, RGB
Class Balance	~98.2% Benign / ~1.8% Malignant
Label Column	`target` — 0 = Benign, 1 = Malignant
Split	70% Train / 15% Validation / 15% Test (stratified)
📥 Download Dataset:
> [https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized)
> **Note:** The dataset is ~4 GB and is **not included** in this repository. Add it as a Kaggle dataset input when running the notebook, or download it manually and place it under `data/` as described in [How to Run](#-how-to-run).
---
🤖 Models
Custom CNN (trained from scratch)
Layer	Output Shape	Parameters
Input	256 × 256 × 3	—
Conv Block 1 (32 filters, Dropout 0.20)	128 × 128 × 32	18,752
Conv Block 2 (64 filters, Dropout 0.20)	64 × 64 × 64	74,048
Conv Block 3 (128 filters, Dropout 0.25)	32 × 32 × 128	295,168
Conv Block 4 (256 filters, Dropout 0.25)	16 × 16 × 256	1,180,160
Conv Block 5 (512 filters, Dropout 0.30)	8 × 8 × 512	4,719,104
Global Average Pooling	512	—
Dense (512) + L2 + Dropout 0.50	512	262,656
Dense (256) + L2 + Dropout 0.40	256	131,328
Output Softmax (2)	2	514
Total		~6.68M
Each conv block: `Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPooling2D → Dropout`
Transfer Learning Models
Model	Total Params	Unfrozen Layers	Phase 1 Best Val Acc	Key Architecture
MobileNetV2	3.4M	Last 50	93.48%	Inverted residuals, depthwise separable conv
EfficientNetB0	5.3M	Last 50	98.23%	Compound depth/width/resolution scaling
ResNet50	25.6M	Last 50	87.36%	50-layer skip-connection residual learning
Shared classification head (all TL models):
```
GlobalAveragePooling2D → BatchNormalization
→ Dense(512, relu, L2=1e-4) → Dropout(0.50)
→ Dense(256, relu, L2=1e-4) → Dropout(0.40)
→ Dense(2, softmax)
```
Two-phase training:
Phase 1 — 10 epochs, lr=1e-3: backbone frozen, head only
Phase 2 — 25 epochs, lr=1e-4: last 50 backbone layers unfrozen
---
📊 Results
Validation Accuracy from Training Logs
Model	Phase 1 Best Val Acc	Phase 2 Best Val Acc	Total Epochs
MobileNetV2	93.48% (epoch 3)	96.16% (epoch 0)	P1: 9, P2: 9
EfficientNetB0	98.23% (epoch 4)	98.23% (epoch 0)	P1: 10, P2: 9
ResNet50	87.36% (epoch 5)	98.23% (epoch 3)	P1: 10, P2: 12
Full test-set metrics (Accuracy, F1, AUC, Sensitivity, Specificity, Inference Time) are saved to `outputs/model_comparison.csv` after notebook execution.
Key Findings
🏆 EfficientNetB0 — best overall: highest AUC, 98.23% validation accuracy, 5.3M parameters
⚡ MobileNetV2 — best edge deployment trade-off: fastest inference, fewest TL model parameters
🐘 ResNet50 — matches EfficientNetB0 at Phase 2 val accuracy but 25.6M parameters (4.8× larger)
📉 Custom CNN — lower malignant-class sensitivity due to no pre-training on a severely imbalanced dataset
> ⚠️ **Clinical priority:** Sensitivity (malignant recall) is the critical metric. A missed malignancy (false negative) carries far higher clinical risk than a false positive.
---
📁 Repository Structure
```
Skin_Lesion_Classification_Pattern_Recognition/
│
├── 📓 notebooks/
│   └── phase2-ml-assignment.ipynb          # Complete implementation notebook
│
├── 📊 figures/
│   ├── class_distribution.png              # Class imbalance bar + pie chart
│   ├── augmented_samples.png               # Augmented training samples grid
│   ├── image_stats.png                     # Width / height / pixel distributions
│   │
│   ├── MobileNetV2_curves.png              # Training & validation curves
│   ├── EfficientNetB0_curves.png
│   ├── ResNet50_curves.png
│   │
│   ├── MobileNetV2_cm.png                  # Normalised confusion matrix
│   ├── EfficientNetB0_cm.png
│   ├── ResNet50_cm.png
│   │
│   ├── MobileNetV2_roc.png                 # ROC curve with AUC
│   ├── EfficientNetB0_roc.png
│   ├── ResNet50_roc.png
│   │
│   ├── gradcam_CustomCNN_benign.png        # Grad-CAM: correct benign
│   ├── gradcam_MobileNetV2_benign.png
│   ├── gradcam_MobileNetV2_malignant.png   # Grad-CAM: correct malignant
│   ├── gradcam_EfficientNetB0_benign.png
│   ├── gradcam_ResNet50_benign.png
│   ├── CustomCNN_gradcam_Correct_Benign.png
│   ├── CustomCNN_gradcam_Incorrect_(FP_FN).png
│   ├── EfficientNetB0_inference_demo.png   # Clinical inference demo
│   │
│   ├── CustomCNN_error_analysis.png        # Error analysis panel
│   ├── MobileNetV2_error_analysis.png
│   ├── EfficientNetB0_error_analysis.png
│   ├── ResNet50_error_analysis.png
│   │
│   └── model_comparison.png               # 4-panel model comparison chart
│
├── 📈 outputs/
│   ├── model_comparison.csv               # Full numeric test-set results
│   ├── EfficientNetB0_log.csv             # Epoch training log — Phase 2
│   ├── EfficientNetB0_p1_log.csv          # Epoch training log — Phase 1
│   ├── MobileNetV2_log.csv
│   ├── MobileNetV2_p1_log.csv
│   ├── ResNet50_log.csv
│   └── ResNet50_p1_log.csv
│
├── 🧠 models/
│   └── README_models.md                   # Download links for .h5 weights
│                                           # (weights too large for GitHub — host on Drive/HuggingFace)
│
├── 📄 docs/
│   └── Phase2_Proposal_Nisarga_BM.pdf     # Submitted proposal document
│
├── 📋 data/
│   └── README_dataset.md                  # Dataset info + download instructions
│
├── README.md                              # This file
└── requirements.txt                       # Python dependencies
```
---
▶️ How to Run
Option A — Kaggle Notebook (Recommended — GPU Included Free)
This notebook was built and tested on Kaggle with GPU acceleration.
Step 1. Open the notebook:
> [https://www.kaggle.com/code/nisargabm/nisargabm-skinlesionclassification](https://www.kaggle.com/code/nisargabm/nisargabm-skinlesionclassification)
Step 2. Click Copy & Edit (top right)
Step 3. Add the dataset — in the right panel under Input:
Click + Add Input → search `nischaydnk/isic-2020-jpg-256x256-resized` → Add
Step 4. Enable GPU — under Session Options (right panel):
Accelerator → select GPU T4 x2 or GPU P100
Step 5. Click Run All from the top menu
Step 6. When complete, download outputs:
`figures_output.zip` — all generated figures
`outputs_export.zip` — CSV logs and model comparison
Expected runtime: 2–4 hours depending on GPU type.
---
Option B — Run Locally
1. Clone the Repository
```bash
git clone https://github.com/NisargaBM/Skin_Lesion_Classification_Pattern_Recognition.git
cd Skin_Lesion_Classification_Pattern_Recognition
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Download the Dataset
Via Kaggle API (fastest):
```bash
# First configure your Kaggle API key:
# Download kaggle.json from https://www.kaggle.com/settings → API → Create New Token
# Place it at ~/.kaggle/kaggle.json and set permissions:
chmod 600 ~/.kaggle/kaggle.json

# Then download:
kaggle datasets download -d nischaydnk/isic-2020-jpg-256x256-resized --path data/
cd data && unzip isic-2020-jpg-256x256-resized.zip && cd ..
```
Via browser (manual):
Visit https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized
Click Download (requires free Kaggle account)
Extract into `data/` so the structure looks like:
```
data/
└── isic-2020-jpg-256x256-resized/
    ├── train-metadata.csv
    └── train-image/
        └── image/
            ├── ISIC_0015719.jpg
            ├── ISIC_0052349.jpg
            └── ...   (33,126 images)
```
4. Update Paths in the Notebook
Open `notebooks/phase2-ml-assignment.ipynb` and edit Section 2:
```python
# Replace these two lines with your local paths:
CSV_PATH = Path('data/isic-2020-jpg-256x256-resized/train-metadata.csv')
IMG_DIR  = Path('data/isic-2020-jpg-256x256-resized/train-image/image')
```
5. Launch and Run
```bash
jupyter notebook notebooks/phase2-ml-assignment.ipynb
# or
jupyter lab notebooks/phase2-ml-assignment.ipynb
```
Run all cells top-to-bottom. Output figures are saved to `figures/`, models to `models/`.
> **GPU strongly recommended.** CPU-only training will be approximately 10–20× slower per epoch.
---
Section-by-Section Guide
Section	What It Does
1	Imports, random seeds (SEED=42), global config (IMG_SIZE=256, BATCH=32)
2	Dataset path detection — update `CSV_PATH` and `IMG_DIR` here for local runs
3	Load and inspect `train-metadata.csv`
4	Parse binary labels (`target`: 0=benign, 1=malignant), attach image filepaths
5	EDA — class distribution bar and pie charts
6	Sample image grid (5 benign + 5 malignant)
7	Image dimension and pixel intensity statistics
8	Stratified 70% / 15% / 15% train / val / test split
9	Compute class weights to handle ~98/2% imbalance
10	Build data generators with augmentation (train) and rescale-only (val/test)
11	Shared utilities: callbacks, `plot_history`, `evaluate_model`, `plot_roc`
12	Custom CNN — build, compile, train, evaluate, save
13	MobileNetV2 — two-phase fine-tuning, evaluate
14	EfficientNetB0 — two-phase fine-tuning, evaluate
15	ResNet50 — two-phase fine-tuning, evaluate
16	Model comparison table + 4-panel bar chart
17	Grad-CAM for all four models (correct + incorrect predictions)
18	Error analysis — confidence distributions, FP/FN breakdown, per-class error rates
19	Save all models as `.h5` files
20	Single-image inference demo with Grad-CAM overlay
21	Final summary printout
---
🖼️ Figures and Outputs
All figures generated by the notebook are committed to `figures/`. Below is the full index.
Dataset Exploration
File	Description
`class_distribution.png`	Class imbalance: ~98.2% benign / ~1.8% malignant
`augmented_samples.png`	16-image grid of augmented training samples
`image_stats.png`	Width, height, and mean pixel intensity distributions
Training Curves
File	Description
`MobileNetV2_curves.png`	Accuracy and loss over all epochs (Phase 1 + Phase 2 merged)
`EfficientNetB0_curves.png`	Same for EfficientNetB0
`ResNet50_curves.png`	Same for ResNet50
Confusion Matrices
File	Description
`MobileNetV2_cm.png`	Normalised confusion matrix on test set
`EfficientNetB0_cm.png`	Same for EfficientNetB0
`ResNet50_cm.png`	Same for ResNet50
ROC Curves
File	Description
`MobileNetV2_roc.png`	Binary ROC curve + AUC
`EfficientNetB0_roc.png`	Same for EfficientNetB0
`ResNet50_roc.png`	Same for ResNet50
Grad-CAM Explainability
File	Description
`gradcam_CustomCNN_benign.png`	CustomCNN — correct benign: Original | Heatmap | Overlay
`gradcam_MobileNetV2_benign.png`	MobileNetV2 — correct benign
`gradcam_MobileNetV2_malignant.png`	MobileNetV2 — correct malignant
`gradcam_EfficientNetB0_benign.png`	EfficientNetB0 — correct benign
`gradcam_ResNet50_benign.png`	ResNet50 — correct benign
`CustomCNN_gradcam_Correct_Benign.png`	CustomCNN — correct benign (second set)
`CustomCNN_gradcam_Incorrect_(FP_FN).png`	CustomCNN — false positive and false negative cases
`EfficientNetB0_inference_demo.png`	Clinical inference: image → heatmap → prediction with confidence
Error Analysis
File	Description
`CustomCNN_error_analysis.png`	Confidence histogram, FP/FN pie, per-class error bars
`MobileNetV2_error_analysis.png`	Same for MobileNetV2
`EfficientNetB0_error_analysis.png`	Same for EfficientNetB0
`ResNet50_error_analysis.png`	Same for ResNet50
Model Comparison
File	Description
`model_comparison.png`	4-panel chart: Accuracy, AUC, F1-Score, Sensitivity across all models
Training Logs (`outputs/`)
File	Description
`EfficientNetB0_p1_log.csv`	Epoch-level: accuracy, val_accuracy, loss, val_loss, lr (Phase 1)
`EfficientNetB0_log.csv`	Same for Phase 2
`MobileNetV2_p1_log.csv`	Same
`MobileNetV2_log.csv`	Same
`ResNet50_p1_log.csv`	Same
`ResNet50_log.csv`	Same
`model_comparison.csv`	Final test-set metrics for all four models
---
📦 Dependencies
```
tensorflow>=2.10.0
scikit-learn>=1.1.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
opencv-python>=4.6.0
Pillow>=9.3.0
kaggle>=1.5.12
jupyter>=1.0.0
```
Save as `requirements.txt` in the repository root. Install with:
```bash
pip install -r requirements.txt
```
> Python 3.9 or 3.10 recommended. TensorFlow 2.12 tested on CUDA 11.8 / cuDNN 8.6.
---
⚠️ Disclaimer
This system is developed strictly for academic research as part of the Pattern Recognition course at the University of Europe for Applied Sciences.
Not clinically validated on external patient cohorts
Must not replace professional dermatological diagnosis or biopsy
All model outputs are decision support only, to be reviewed by a qualified clinician
Binary classification only — does not distinguish between specific lesion subtypes (e.g. melanoma vs. BCC)
---
📎 Links
Resource	URL
🔗 Kaggle Notebook	kaggle.com/code/nisargabm/nisargabm-skinlesionclassification
📥 Dataset	kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized
🏥 ISIC Archive	isic-archive.com
---
<div align="center">
<sub>Nisarga B M · M.Sc. Software Engineering · University of Europe for Applied Sciences · 2026</sub>
</div>

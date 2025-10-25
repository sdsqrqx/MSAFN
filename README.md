# MSAFN: Multi-Scale Attention Fusion Network for Ulcerative Colitis Severity Grading

> Official implementation of **MSAFN** (with LSCHL and PI loss) for ulcerative colitis (UC) endoscopic severity grading.

------

## Contents

- [Overview](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#overview--简介)
- [Installation](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#installation--环境安装)
- [Data](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#data--数据)
- [Quick Start](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#quick-start--快速开始)
- [Reproducibility](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#reproducibility--复现实验清单)
- [Model Zoo](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#model-zoo--模型与权重)
- [Results](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#results--主要结果)
- [Ethics & Consent](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#ethics--consent--伦理与同意)
- [Code and Data Availability](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#code-and-data-availability--代码与数据可用性)
- [Cite](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#cite--引用)
- [License](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#license--许可)
- [Contact](https://chatgpt.com/g/g-p-68f753f01ee4819196659def58303dd7-venus/c/68fc26b6-6cc8-8320-a213-a9497f8644f6#contact--联系)

------

## Overview

**MSAFN** addresses the challenges of low contrast, subtle morphology, and label ambiguity in UC endoscopy.
 Core components:

- **Backbone**: EfficientNet-based feature pyramid.
- **LSCHL**: Lesion-guided spatial–channel–hierarchical learning (channel–frequency reweighting + prototype-prior spatial routing + prior-gated, offset-aware cross-scale injection).
- **PI loss**: Progressive inter-class separation loss that shifts from representation learning to discriminative optimization over time.

------

## Installation

Tested on: **Python 3.10**, **PyTorch 2.2+**, **CUDA 11.8+** (Linux/Windows).

```bash
# 1) Create env
conda create -n msafn python=3.10 -y
conda activate msafn

# 2) Install deps
pip install -r requirements.txt
# or
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations tqdm pyyaml pandas scikit-learn einops timm matplotlib

# 3) (Optional) Reproduce exact env
conda env create -f environment.yml
```

------

## Data

### Public datasets

- **HyperKvasir** — public UC-related endoscopic dataset.
- **LIMUC**  — public UC dataset.
   Download from their official websites (or mirrors) and place under `data/`. We provide scripts to standardize structure and verify checksums.

```bash
# Example directory layout
data/
├─ hyperkvasir/
│  ├─ images/ ... 
│  └─ labels.csv
└─ limuc/
   ├─ images/ ...
   └─ labels.csv
```

- Checksums & links: see `data/README_DATA.md` (we provide MD5/SHA256 and download instructions).
- Licenses: follow each dataset’s original license.

### Institutional dataset (controlled) 

- **LHUCE** (Lianyungang Hospital UC Endoscopy): **supplementary validation only**.
  - Origin: Second People’s Hospital of Lianyungang.
  - Status: fully anonymized; **not publicly released**; available under **controlled access**.
  - Access: contact **[Name]** (**[email@hospital]**); requires a Data Use Agreement (DUA).

> We **do not** distribute any patient-identifiable information.

### Prepare splits 

We provide exact train/val/test splits (CSV) and preprocessing scripts:

```bash
python tools/prepare_splits.py \
  --dataset hyperkvasir --root data/hyperkvasir --out configs/splits/hyperkvasir/
python tools/prepare_splits.py \
  --dataset limuc --root data/limuc --out configs/splits/limuc/
# LHUCE (supplementary): prepare local CSVs but not committed to repo
```

------

## Quick Start

### 1) Training 

```bash
# HyperKvasir
python train.py --config configs/hyperkvasir_msafn.yaml \
  TRAIN.SEED 2024 TRAIN.BATCH_SIZE 32 TRAIN.EPOCHS 120

# LIMUC
python train.py --config configs/limuc_msafn.yaml \
  TRAIN.SEED 2024 TRAIN.BATCH_SIZE 32 TRAIN.EPOCHS 120
```

### 2) Evaluation 

```bash
python eval.py --config configs/hyperkvasir_msafn.yaml --ckpt weights/hyperkvasir_msafn.pth
python eval.py --config configs/limuc_msafn.yaml --ckpt weights/limuc_msafn.pth
# LHUCE (supplementary validation)
python eval.py --config configs/lhuce_msafn.yaml --ckpt weights/hyperkvasir_msafn.pth
```

### 3) Inference

```bash
python infer.py --ckpt weights/hyperkvasir_msafn.pth --img path/to/image.jpg --out results/demo.png
```

------

## Reproducibility

To fully reproduce tables and figures in the paper:

1. **Environment lock**
   - Use `environment.yml` or `pip-tools` frozen `requirements.txt`.
   - GPU/driver info saved via `tools/print_env.py`.
2. **Determinism**
   - Global seed set in configs (`TRAIN.SEED`), torch/cuDNN flags fixed.
3. **Exact splits**
   - Train/val/test CSVs under `configs/splits/*` (with image list and labels).
   - For third-party datasets, we provide preprocessing to reproduce our splits.
4. **Config & logs**
   - All hyperparameters stored in `configs/*.yaml`.
   - `results/` contains training logs, curves, confusion matrices.
5. **Checksums**
   - Dataset archive/file checksums in `data/README_DATA.md`.
6. **One-click scripts**
   - `scripts/run_all.sh` will run training + evaluation to reproduce Tables **[X]** and Figures **[Y]**.

------

## Results 

Key metrics (mean ± std over 3 runs, seed = 2023/2024/2025).

- HyperKvasir: **[AUROC/Acc/F1]**
- LIMUC: **[AUROC/Acc/F1]**
- LHUCE (supplementary only): **[Acc/F1]**

Full tables and plots are auto-generated to `results/` via:

```bash
bash scripts/run_all.sh
python tools/plot_curves.py --logdir results/
```

------

## Ethics & Consent 

- **LHUCE** images were **fully anonymized** before transfer (names, IDs, facial features, metadata removed).
- Retrospective selection from routine clinical examinations; analysis limited to lesion areas.
- According to the Second People’s Hospital of Lianyungang internal policy, research using fully anonymized retrospective imaging without patient contact/intervention is **exempt from formal ethics committee approval**; written informed consent was **waived**.
- The study complies with the **Declaration of Helsinki**.

------

## Code and Data Availability 

- This study primarily uses **public** UC datasets: **HyperKvasir** [49] and **LIMUC** [48] under their respective licenses.
- Our code, configs, preprocessing and evaluation scripts are publicly available in this repository.
- An additional institutional dataset **LHUCE** (Second People’s Hospital of Lianyungang) was used **only for supplementary validation**.
  - LHUCE is **not publicly released** due to institutional data-use restrictions despite full anonymization.
  - Qualified researchers may request **controlled access** from the corresponding author and the hospital’s data access committee (contact: **[Name]**, **[email]**), subject to a **Data Use Agreement**.
- We **do not** distribute any patient-identifiable information.

------

## Cite / 引用

If you find this repo useful, please cite:

```bibtex
@article{YourMSAFN2025,
  title   = {MSAFN: A Multi-Scale Attention Fusion Network for Ulcerative Colitis Severity Grading in Endoscopic Images},
  author  = {Your Name and Coauthors},
  journal = {Journal},
  year    = {2025},
  doi     = {10.xxxx/xxxxx},
  url     = {https://github.com/sdsqrqx/MSAFN}
}
```

------

## License

- Code license: **[MIT/Apache-2.0/BSD-3-Clause/GPL-3.0 – choose one]** (see `LICENSE`).
- Pretrained weights are released for **research use**; redistribution must follow the dataset and model licenses.
- Public datasets **HyperKvasir/LIMUC** follow their original licenses.

------

## Contact 

- Issues & PRs are welcome: please open a GitHub Issue describing the bug/feature with logs and config.

------

### Acknowledgments

We thank the maintainers of **HyperKvasir** and **LIMUC**, and the clinical team of the **Second People’s Hospital of Lianyungang** for data curation (LHUCE).

------

### Changelog 

- **v1.0.0** – Initial public release: training/eval, configs, scripts, splits for HyperKvasir & LIMUC; supplementary LHUCE eval support.
- **v1.0.1** – [Add bugfix/feature].

------

### FAQ

- **Can I get LHUCE images?**
   LHUCE is under controlled access; contact the corresponding author and hospital DAC, subject to a DUA.
- **Where are exact splits?**
   See `configs/splits/*`. We also provide scripts to regenerate them from raw downloads.


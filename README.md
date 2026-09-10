# Seeing in the Dark

Does cleaning up a dark photo actually help a computer see? This project benchmarks
classical (OpenCV) and generative (AE / VAE / cGAN / CycleGAN) low-light restoration
against YOLO detection accuracy on LoLI-Street and ExDark.

## Hypothesis

Object detectors trained on well-lit images lose accuracy on low-light images.
Classical DSP-based restoration, paired generative restoration, and unpaired
generative restoration can each recover some of that lost accuracy — but by how
much, where does each fail, and does the answer change once the detector itself
is allowed to adapt to low light?

## Datasets (local only, see `data_manifests/`)

| Dataset      | Type                                    | Used for                                              |
|--------------|-----------------------------------------|-------------------------------------------------------|
| LoLI-Street  | 30k paired low/well-lit street images   | Classical tuning, AE/VAE/cGAN training, PSNR/SSIM eval |
| ExDark       | 7,363 real unpaired low-light images    | Zero-shot eval, CycleGAN training, mAP-only eval      |

## Pipeline

- **Part A** — classical DSP restoration (`src/classical/`): CLAHE, gamma, hand-implemented Retinex, spatial + frequency denoising, morphology.
- **Part B** — paired generative models from scratch (`src/paired/`): convolutional AE, VAE, pix2pix-style cGAN.
- **Part C** — unpaired CycleGAN (`src/cyclegan/`): ExDark low-light ↔ LoLI-Street well-lit.
- **YOLO spine** (`src/yolo_eval/`): one harness evaluates every (dataset, restoration, detector) combination; YOLOv8 zero-shot plus fine-tunes on each dataset.
- **Master table** (`notebooks/05_master_table_analysis.ipynb`): does PSNR/SSIM track mAP, or diverge?

## Status

Work in progress, built incrementally — see commit history.
Each notebook in `notebooks/` is numbered in build order.

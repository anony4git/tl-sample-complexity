# tl-sample-complexity

## TL_Data

It is a dataset demonstrating the sample complexty of transfer learning that contains data from three different domains:
- **Amazon**: Product images downloaded from the Amazon website  
- **DSLR**: Images taken with a digital single-lens reflex camera  
- **Webcam**: Images taken with a webcam  

### Dataset Statistics

#### Overall Statistics
| Dataset | Number of Samples | Proportion |
|---------|-------------------|------------|
| Amazon  | 2,817             | 68.5%      |
| DSLR    | 498               | 12.1%      |
| Webcam  | 795               | 19.4%      |
| **Total** | **4,110**       | **100%**   |

#### Category Distribution
The dataset contains 31 item categories, with the following distribution across the three domains:

| Category Name | Amazon | DSLR | Webcam | Total |
|---------------|--------|------|--------|-------|
| back_pack | 92 | 12 | 29 | 133 |
| bike | 82 | 21 | 21 | 124 |
| bike_helmet | 72 | 24 | 28 | 124 |
| bookcase | 82 | 12 | 12 | 106 |
| bottle | 36 | 16 | 16 | 68 |
| calculator | 94 | 12 | 31 | 137 |
| desk_chair | 91 | 13 | 40 | 144 |
| desk_lamp | 97 | 14 | 18 | 129 |
| desktop_computer | 97 | 15 | 21 | 133 |
| file_cabinet | 81 | 15 | 19 | 115 |
| headphones | 99 | 13 | 27 | 139 |
| keyboard | 100 | 10 | 27 | 137 |
| laptop_computer | 100 | 24 | 30 | 154 |
| letter_tray | 98 | 16 | 19 | 133 |
| mobile_phone | 100 | 31 | 30 | 161 |
| monitor | 99 | 22 | 43 | 164 |
| mouse | 100 | 12 | 30 | 142 |
| mug | 94 | 8 | 27 | 129 |
| paper_notebook | 96 | 10 | 28 | 134 |
| pen | 95 | 10 | 32 | 137 |
| phone | 93 | 13 | 16 | 122 |
| printer | 100 | 15 | 20 | 135 |
| projector | 98 | 23 | 30 | 151 |
| punchers | 98 | 18 | 27 | 143 |
| ring_binder | 90 | 10 | 40 | 140 |
| ruler | 75 | 7 | 11 | 93 |
| scissors | 100 | 18 | 25 | 143 |
| speaker | 99 | 26 | 30 | 155 |
| stapler | 99 | 21 | 24 | 144 |
| tape_dispenser | 96 | 22 | 23 | 141 |
| trash_can | 64 | 15 | 21 | 100 |

### Dataset Feature Analysis

#### 1. Amazon Dataset
- **Number of Samples**: 2,817 images  
- **Characteristics**:
  - Largest number of samples, accounting for 68.5% of the total  
  - Relatively high image quality, clean backgrounds  
  - Relatively standardized product display angles  
- **Advantages**: Large dataset, suitable as a source domain for pre-training  
- **Disadvantages**: Large differences from real-world shooting environments  

#### 2. DSLR Dataset
- **Number of Samples**: 498 images  
- **Characteristics**:
  - Smallest number of samples, only 12.1% of the total  
  - Captured with professional cameras, highest image quality  
- **Advantages**: High quality images, rich details  
- **Disadvantages**: Small dataset size, potential risk of overfitting  

#### 3. Webcam Dataset
- **Number of Samples**: 795 images  
- **Characteristics**:
  - Medium-sized dataset, accounting for 19.4% of the total  
  - Captured with webcams, relatively lower image quality  
- **Advantages**: Closer to real-world application scenarios  
- **Disadvantages**: Relatively lower image quality, more noise  

### Category Imbalance Analysis

#### Sample Count Distribution
- **Most Samples**: monitor (164 images)  
- **Fewest Samples**: bottle (68 images)  
- **Average Samples per Category**: 132.6 images  

#### Inter-Domain Distribution Differences
1. **Amazon Domain**: Most categories have 90–100 samples, relatively balanced  
2. **DSLR Domain**: Sample counts range from 7–31, larger variation  
3. **Webcam Domain**: Sample counts range from 11–43, relatively balanced  

## 31-Class Transfer Learning Experiment Framework

This is a complete framework for multi-dataset transfer learning experiments, supporting various initialization modes, automated grid search, and detailed performance evaluation.

### 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Parameter Description](#parameter-description)
- [Experiment Modes](#experiment-modes)
- [Output Description](#output-description)
- [Advanced Features](#advanced-features)
- [Examples](#examples)

### ✨ Features

#### Core Features
- **Multiple Initialization Modes**: Support for ImageNet pretraining, random initialization, source domain pretraining, etc.
- **Flexible Data Splitting**: Fixed 10% test set, customizable training set ratio
- **Grid Search Experiments**: Automated batch experiments, supporting multiple configuration combinations
- **Early Stopping Mechanism**: Pretraining phase supports validation set and early stopping to avoid overfitting
- **Mixed Precision Training**: Uses AMP to accelerate training and reduce memory usage
- **Detailed Performance Evaluation**: Including accuracy, recall, F1, AUROC and other metrics
- **Enhanced Classification Report**: Per-class specificity, balanced accuracy, OVR-AUC, etc.

#### Data Processing
- Stratified sampling to ensure class balance
- Support for merging multiple datasets
- Automatic class mapping alignment
- Data augmentation (random flip, rotation)

#### Training Optimization
- AdamW optimizer + Cosine annealing learning rate
- Automatic mixed precision training
- Asynchronous result saving
- GPU acceleration (CUDA support)

### 🔧 Requirements

#### Dependencies
```bash
torch
torchvision
numpy
scikit-learn
pandas
```


### 🚀 Quick Start

#### 1. Configure Datasets

First configure dataset paths in `config.py`:

```python
DATASETS_MAP = {
    "dataset1": "/path/to/dataset1",
    "dataset2": "/path/to/dataset2",
    "dataset3": "/path/to/dataset3",
}
```

#### 2. Run Basic Experiment

```bash
# Activate conda environment
conda activate your_environment

# Run with default configuration
python main.py

# Custom configuration
python main.py \
    --fixed_source dataset1 \
    --splits 0.1 0.3 0.5 \
    --init_modes resnet50_imagenet random \
    --epochs_finetune 10 20 30 \
    --batch_size 32
```

#### 3. Generate Experiment Summary

```bash
# Generate CSV summary from JSON files
python main.py --summary --out_dir exp_out
```

### 📖 Parameter Description

#### Dataset Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--fixed_source` | str | First dataset | Fixed source dataset for pretraining |
| `--target_mode` | str | `merged` | Target mode: `merged` or `separate` |

#### Training Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--splits` | float[] | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` | Training set ratio list (based on training set) |
| `--epochs_finetune` | int[] | `[100]` | Fine-tuning epochs list |
| `--epochs_pretrain` | int | `50` | Pretraining epochs |
| `--batch_size` | int | `32` | Batch size |
| `--lr` | float | `0.001` | Learning rate |
| `--img_size` | int | `224` | Image size |

#### Model Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--init_modes` | str[] | `[resnet50_imagenet, random]` | Initialization mode list |
| `--no_amp` | flag | False | Disable mixed precision training |

#### Early Stopping Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--val_split` | float | `0.2` | Pretraining validation split ratio |
| `--patience` | int | `5` | Early stopping patience value |

#### Other Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--seed` | int | `42` | Random seed |
| `--num_workers` | int | `4` | Number of data loading threads |
| `--out_dir` | str | `exp_out` | Output directory |
| `--summary` | flag | False | Only generate summary file |

### 🎯 Experiment Modes

#### 1. resnet50_imagenet
Use ImageNet pretrained ResNet50 model, directly fine-tune on target domain.

**Use Case**: Target domain similar to natural images

```bash
python main.py --init_modes resnet50_imagenet
```

#### 2. random
Completely random initialization, train from scratch.

**Use Case**: Comparison baseline, understand pretraining value

```bash
python main.py --init_modes random
```

#### 3. random_then_fixed_src
After random initialization, first pretrain on fixed source domain, then fine-tune on target domain.

**Use Case**: Strong correlation between source and target domains

```bash
python main.py \
    --init_modes random_then_fixed_src \
    --fixed_source dataset1 \
    --epochs_pretrain 50
```

### 📊 Output Description

#### File Structure

```
exp_out/
├── source_to_target_resnet50_imagenet_r01_e10.json
├── source_to_target_resnet50_imagenet_r01_e20.json
├── source_to_target_random_r03_e10.json
└── summary.csv
```

#### JSON File Format

Each experiment generates a JSON file containing:

```json
{
  "target_datasets": ["dataset1", "dataset2"],
  "split": 0.5,
  "init_mode": "resnet50_imagenet",
  "source_dataset": "None",
  "epochs_finetune": 20,
  "acc": 0.8523,
  "recall": 0.8456,
  "precision": 0.8589,
  "f1_macro": 0.8521,
  "f1_micro": 0.8523,
  "auroc": 0.9234,
  "cls_report": "..."
}
```

#### Metrics Description

- **acc**: Overall accuracy
- **recall**: Macro-averaged recall
- **precision**: Macro-averaged precision
- **f1_macro**: Macro-averaged F1 score
- **f1_micro**: Micro-averaged F1 score
- **auroc**: Macro-averaged AUROC (OVR)
- **cls_report**: Detailed per-class report (including specificity, balanced accuracy, etc.)

#### Summary CSV

Contains summary information of all experiments, sorted by:
1. target_datasets
2. init_mode
3. split
4. epochs_finetune


### 📝 Examples

#### Example 1: Complete Grid Search

```bash
python main.py \
    --fixed_source dataset1 \
    --splits 0.1 0.3 0.5 0.7 0.9 \
    --init_modes resnet50_imagenet random random_then_fixed_src \
    --epochs_finetune 10 20 30 40 50 \
    --epochs_pretrain 50 \
    --batch_size 32 \
    --lr 0.001 \
    --target_mode merged
```

#### Example 2: Quick Test

```bash
python main.py \
    --splits 0.5 \
    --init_modes resnet50_imagenet \
    --epochs_finetune 10 \
    --batch_size 64
```

#### Example 3: Pretraining Experiment

```bash
python main.py \
    --init_modes random_then_fixed_src \
    --fixed_source dataset1 \
    --epochs_pretrain 100 \
    --val_split 0.2 \
    --patience 10 \
    --splits 0.5 0.9
```

#### Example 4: Transfer to Multiple Targets Separately

```bash
python main.py \
    --target_mode separate \
    --fixed_source dataset1 \
    --splits 0.5 \
    --init_modes resnet50_imagenet random
# Will transfer to dataset2 and dataset3 separately
```




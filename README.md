# FCM
"[Improving Model Robustness With Frequency Component Modification and Mixing](https://ieeexplore.ieee.org/document/10776988)"

## Contents

This project supports CIFAR-10, CIFAR-100, and ImageNet datasets. Evaluation can be performed on their corresponding corruption benchmarks: CIFAR-10-C, CIFAR-100-C, and ImageNet-C.

Freqtune == FCM

## Usage

🏋️‍♂️ Training

1. Train on CIFAR-10 (default)
```
python cifar.py
```

2. Train on CIFAR-100
```
python cifar.py --dataset cifar100
```

📊 Evaluation

1. Evaluate a trained CIFAR-10 model
```
python cifar.py --resume <path_to_model> --evaluate
# Example:
python cifar.py --resume ../FCM/snapshots/model_best.pth.tar --evaluate
```

2. Evaluate a trained CIFAR-100 model
```
python cifar.py --resume <path_to_model> --evaluate --dataset cifar100
# Example:
python cifar.py --resume ../FCM/snapshots/model_best.pth.tar --evaluate --dataset cifar100
```

# FCM
"[Improving Model Robustness With Frequency Component Modification and Mixing](https://ieeexplore.ieee.org/document/10776988)"

"This repository is based on [PixMix](https://github.com/andyzoujm/pixmix), with several modifications and extensions."

## Contents

This project supports CIFAR-10, CIFAR-100, and ImageNet datasets. Evaluation can be performed on their corresponding corruption benchmarks: CIFAR-10-C, CIFAR-100-C, and ImageNet-C.

* Freqtune == FCM
* ImageNet 사용 시 data파일에 추가 다운로드 필요 

1. data_path 변경
``` cifar.py
parser.add_argument('--data_path', type=str, *default='/home/hyunji/Documents/FreqTune/data'*, required=False, help='Path to CIFAR and CIFAR-C directories')
```

2. FCM 비율은 여기서 조절
``` cifar.py
parser.add_argument('--p', default=0.5, type=float, help='Random Frequency region, FreqTune')
```

3. 구체적인 FCM 동작은 여기서 조절
## FreqTune_transform.py
```
        height = 32
        width = 32        # cifar 이미지이므로 32x32로 고정
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)      # DFT(FCM은 shift진행x, 따라서 중앙이 고주파수 외곽이 저주파수)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기(논문 내용대로)
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 고주파수 영역 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # 저주파수 강도
        B = 0.5
        b = np.random.uniform(0, B)
        array2 = np.random.uniform(1-b, 1+b, size=fft_1.shape)

        # 고주파수 강도
        A = 5
        a = np.random.uniform(0, A)
        array1 = np.random.uniform(-a, a, size=matrix.shape)

        # 행렬곱, transform part
        fft_1 = fft_1 * array2
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        # IDFT
        img = np.fft.ifftn(fft_1)
        new_image = np.clip(img, 0, 255).astype(np.uint8)      # 픽셀 뒤집힘 방지하기 위해 clip
        x = Image.fromarray(new_image)
        return x
```

4. 모델 train 후 결과는 snapshots 폴더!


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

import random
import numpy as np
from PIL import Image


class FCM(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32  # cifar 이미지이므로 32x32로 고정
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)  # DFT(FCM은 shift진행x, 따라서 중앙이 고주파수 외곽이 저주파수)

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
        array2 = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)

        # 고주파수 강도
        A = 5
        a = np.random.uniform(0, A)
        array1 = np.random.uniform(-a, a, size=matrix.shape)

        # 행렬곱, transform part
        fft_1 = fft_1 * array2
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        # IDFT
        img = np.fft.ifftn(fft_1)
        new_image = np.clip(img, 0, 255).astype(np.uint8)  # 픽셀 뒤집힘 방지하기 위해 clip
        x = Image.fromarray(new_image)
        return x
import numpy as np
from PIL import Image

test_img = '/home/hyunji/Downloads/train/dog/blenheim_spaniel_s_000317.png'
x = Image.open(test_img)

A = 5
B = 0.5
n = 32
img = np.array(x).astype(np.uint8)
fft_1 = np.fft.fftn(img)

# 새로운 행렬 만들기
array = np.zeros((n, n), dtype=float)
center = (n // 2, n // 2)

# 중심에서의 최대 거리 계산
max_distance = np.sqrt((center[0]) ** 2 + (center[1]) ** 2)

# 각 요소에 값 할당
for i in range(n):
    for j in range(n):
        # 중심으로부터의 거리 계산
        distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
        # 거리 정규화 (0 ~ 1 사이)
        normalized_distance = distance / max_distance
        # 제곱 함수에 기반한 비선형 감소
        square_distance = normalized_distance ** 2
        # a와 b 사이의 값으로
        value = B + (A - B) * (1 - square_distance)
        array[i, j] = value

# 세 채널에 동일한 값 복사
array_3d = np.dstack((array, array, array))
# 행렬곱, 다시 넣기
fft_1 = fft_1 * array_3d

x = np.fft.ifftn(fft_1)
y = 20 * np.log(np.abs(fft_1))  #####

# new_x = np.clip(x, 0, 255).astype(np.uint8)
# new_y = np.clip(y, 0, 255).astype(np.uint8)
new_x = x.astype(np.uint8)
new_y = y.astype(np.uint8)
# new_x = np.clip(x.real, 0, 255).astype(np.uint8)
# new_y = np.clip(y.real, 0, 255).astype(np.uint8)

xx = Image.fromarray(new_x)
yy = Image.fromarray(new_y)

xx.show()
# yy.show()


# # 255보다 큰 요소들의 위치를 찾기
# high_indices = np.where(new_x > 255)
# for idx in zip(*high_indices):
#     print(idx, "is high")
#
# # 0보다 작은 요소들의 위치를 찾기
# low_indices = np.where(new_x < 0)
# for idx in zip(*low_indices):
#     print(idx, "is low")
#
# np.set_printoptions(threshold=np.inf)
# print(new_x)
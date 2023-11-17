import torch
import numpy as np

sequential_array = np.arange(1, 26).reshape((5, 5, 1))

# 패딩 적용
padded_array = np.pad(sequential_array, ((2, 2), (2, 2), (0, 0)), mode='edge')

# 결과를 저장할 리스트 생성
result_list = []

# 중앙을 기준으로 3x3 윈도우에서 값을 추출하여 리스트에 추가
for i in range(2, padded_array.shape[0] - 2):
    result_x = []
    for j in range(2, padded_array.shape[1] - 2):
        window = padded_array[i-2:i+3, j-2:j+3, 0]  # 3x3 윈도우 추출
        central_element = window.reshape(25,)  # 중앙 원소 추출
        result_x.append(central_element)
    result_list.append(result_x)
    

# 결과 출력
result_array = np.array(result_list)
print(result_array.shape)
print(result_array[0,2])

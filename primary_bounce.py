import os
import exr
import numpy as np
import torch
import time
random_path = "/home/cglab/Desktop/Path-Graph-Convolution/rand_scene1_bounce17.xml"

# 이미지는 x, y로 저장됨

start = time.time()

os.system('rm /home/cglab/Desktop/Path-Graph-Convolution/path_data.txt')

os.system('~/Desktop/path_feature/build/mitsuba {}'.format(random_path))
os.system('cp /home/cglab/Desktop/Path-Graph-Convolution/rand_scene1_bounce17.exr /home/cglab/Desktop/Path-Graph-Convolution/image.exr')
os.system('rm /home/cglab/Desktop/Path-Graph-Convolution/rand_scene1_bounce17.exr')

# os.system('cp /home/cglab/Desktop/Path-Graph-Convolution/path_data_{}.txt /home/cglab/Desktop/Path-Graph-Convolution/path_data_{}.txt'.format(1,1))
# os.system('rm /home/cglab/Desktop/Path-Graph-Convolution/path_data_{}.txt'.format(1,1))

shape = (720, 1280, 2)


# # 각 값이 [0, 0]부터 [719, 1199]까지인 3차원 배열 생성
array = np.empty(shape, dtype=int)

for i in range(shape[1]):
    for j in range(shape[0]):
        for k in range(shape[2]):
            array[j, i, k] = [i, j][k]
exr.write("index.exr", array)

end = time.time()
print(str(end-start) + " == evaluation time")



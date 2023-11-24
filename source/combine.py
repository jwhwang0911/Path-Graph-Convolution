import exr
from utils import tensor2img


noisy = exr.read("/home/jangwon/Desktop/Path-Graph-Convolution/noisy.exr")
gt = exr.read("/home/jangwon/Desktop/Path-Graph-Convolution/gt.exr")
albedo = exr.read("/home/jangwon/Desktop/Path-Graph-Convolution/albedo.exr")
depth = exr.read("/home/jangwon/Desktop/Path-Graph-Convolution/depth.exr")
normal = exr.read("/home/jangwon/Desktop/Path-Graph-Convolution/normal.exr")

print(albedo.shape)
print(depth.shape)
print(normal.shape)

exr.write("/home/jangwon/Desktop/Path-Graph-Convolution/input.exr",
          {
              "noisy" : noisy,
              "albedo" : albedo[:,:,:3],
              "depth" : depth,
              "normal" : normal,
          }
          )

exr.write("/home/jangwon/Desktop/Path-Graph-Convolution/albedo.exr",albedo[:,:,:3])
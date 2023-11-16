import exr

noisy = exr.read("image.exr")
aux = exr.read_all("feature.exr")
albedo = exr.read("feature.exr", "dd")
print(albedo.shape)
normal = aux['nn']
depth = exr.read("feature.exr", "dd.y.T")
# normal
exr.write(
    "noisy.exr",
    {
        "noisy" : noisy,
        "albedo" : albedo[:,:,:3],
        "normal" : normal,
        "depth" : depth,

    }
)

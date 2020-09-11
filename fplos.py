from torchstat import stat
import CBW

n_bands = 200
n_classes = 16
patch_size = 11
model = CBW.MODEL(n_bands, n_classes, patch_size=patch_size)  # 模型
stat(model, (n_bands, patch_size, patch_size))
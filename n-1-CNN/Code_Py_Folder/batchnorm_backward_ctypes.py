import ctypes
import numpy as np
import tensorflow as tf

EPSILON = 1e-5

# ==== Load thư viện C đã biên dịch ====
lib = ctypes.CDLL("n-1-CNN/myfunction.so")

# ==== Khai báo kiểu dữ liệu cho hàm C ====
lib.batchnorm_backward.restype = None
lib.batchnorm_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # dL_dy
    ctypes.c_float,                  # learning_rate
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.POINTER(ctypes.c_float),  # mean
    ctypes.POINTER(ctypes.c_float),  # variance
    ctypes.c_int,                    # batch_size
    ctypes.POINTER(ctypes.c_float),  # output (dL_dxi)
    ctypes.c_int,                    # input_width
    ctypes.c_int,                    # input_height
    ctypes.c_int                     # input_channels
]

# ==== Tham số ====
batch_size = 2
input_h, input_w, input_c = 2, 2, 1
m = batch_size * input_h * input_w

# ==== Dữ liệu mẫu ====
np.random.seed(42)
input_np = np.random.rand(batch_size, input_h, input_w, input_c).astype(np.float32)
dL_dy_np = np.random.rand(batch_size, input_h, input_w, input_c).astype(np.float32)

mean_np = np.mean(input_np, axis=(0,1,2)).astype(np.float32)
var_np = np.var(input_np, axis=(0,1,2)).astype(np.float32)
gamma_np = np.ones(input_c, dtype=np.float32)
beta_np = np.zeros(input_c, dtype=np.float32)

output_c = np.zeros_like(input_np)

# ==== Chuyển sang con trỏ C ====
input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dL_dy_ptr = dL_dy_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
gamma_ptr = gamma_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
beta_ptr = beta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
mean_ptr = mean_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
var_ptr = var_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ==== Gọi hàm C ====
lib.batchnorm_backward(
    input_ptr, dL_dy_ptr, ctypes.c_float(0.01),
    gamma_ptr, beta_ptr,
    mean_ptr, var_ptr,
    batch_size, output_ptr,
    input_w, input_h, input_c
)

# ==== Tính gradient thủ công trong TensorFlow ====
x = input_np
dy = dL_dy_np
mean = mean_np
var = var_np
gamma = gamma_np

# (x - mean) / sqrt(var + eps)
x_norm = (x - mean) / np.sqrt(var + EPSILON)

# dL/dx = (1 / sqrt(var + eps)) * (dy * gamma - mean(dy * gamma) - x_norm * mean(dy * gamma * x_norm))
dy_gamma = dy * gamma
mean_dy_gamma = np.mean(dy_gamma, axis=(0, 1, 2), keepdims=True)
mean_dy_gamma_xnorm = np.mean(dy_gamma * x_norm, axis=(0, 1, 2), keepdims=True)

dx = (1.0 / np.sqrt(var + EPSILON)) * (
    dy_gamma - mean_dy_gamma - x_norm * mean_dy_gamma_xnorm
)

# ==== So sánh kết quả ====
print("✅ Grad Input khớp TensorFlow:", np.allclose(output_c, dx, atol=1e-5))

print("\n📊 Sai lệch Grad Input:")
print(np.round(output_c - dx, 5))

print("\n✅ Grad Input từ C:")
print(output_c.reshape(batch_size, input_h, input_w, input_c))

print("\n✅ Grad Input thủ công từ TensorFlow:")
print(dx.reshape(batch_size, input_h, input_w, input_c))

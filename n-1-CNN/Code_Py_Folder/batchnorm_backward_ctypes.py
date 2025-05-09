import ctypes
import numpy as np
import tensorflow as tf

EPSILON = 1e-5
lr = 0.01

# ==== Load thư viện C đã biên dịch ====
lib = ctypes.CDLL("n-1-CNN/myfunction.so")

# ==== Khai báo kiểu dữ liệu cho hàm C ====
lib.batchnorm_backward.restype = None
lib.batchnorm_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # dL_dy
    ctypes.c_float,                  # learning_rate
    ctypes.POINTER(ctypes.c_float),  # gamma (updated inside)
    ctypes.POINTER(ctypes.c_float),  # beta  (updated inside)
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

# Sao lưu gamma, beta để dùng cho TF và NumPy
gamma_before = gamma_np.copy()
beta_before = beta_np.copy()

output_c = np.zeros_like(input_np)

# ==== Con trỏ C ====
input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dL_dy_ptr = dL_dy_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
gamma_ptr = gamma_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
beta_ptr = beta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
mean_ptr = mean_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
var_ptr = var_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ==== Gọi hàm C ====
lib.batchnorm_backward(
    input_ptr, dL_dy_ptr, ctypes.c_float(lr),
    gamma_ptr, beta_ptr,
    mean_ptr, var_ptr,
    batch_size, output_ptr,
    input_w, input_h, input_c
)

# ==== TensorFlow Tính toán ====
x_tf = tf.constant(input_np)
dy_tf = tf.constant(dL_dy_np)
gamma_tf = tf.constant(gamma_before)
beta_tf = tf.constant(beta_before)

__, mean_tf, var_tf = tf.compat.v1.nn.fused_batch_norm(
    x_tf, gamma_tf, beta_tf, epsilon=EPSILON, is_training=True
)

dx_tf, dgamma_tf, dbeta_tf, _, _ = tf.raw_ops.FusedBatchNormGrad(
    y_backprop=dy_tf,
    x=x_tf,
    scale=gamma_tf,
    reserve_space_1=mean_tf,
    reserve_space_2=var_tf,
    epsilon=EPSILON,
    is_training=True
)

dx_tf_np = dx_tf.numpy()
dgamma_tf_np = dgamma_tf.numpy()
dbeta_tf_np = dbeta_tf.numpy()

# ==== Tính "chay" bằng NumPy ====
x = input_np
dy = dL_dy_np
mean = mean_np
var = var_np
gamma = gamma_before

x_norm = (x - mean) / np.sqrt(var + EPSILON)
dy_gamma = dy * gamma
mean_dy_gamma = np.mean(dy_gamma, axis=(0,1,2), keepdims=True)
mean_dy_gamma_xnorm = np.mean(dy_gamma * x_norm, axis=(0,1,2), keepdims=True)
dx_np = (1.0 / np.sqrt(var + EPSILON)) * (dy_gamma - mean_dy_gamma - x_norm * mean_dy_gamma_xnorm)

dgamma_np = np.sum(dy * x_norm, axis=(0,1,2))
dbeta_np = np.sum(dy, axis=(0,1,2))

# ==== Cập nhật gamma, beta bằng NumPy và TF ====
gamma_tf_updated = gamma_before - lr * dgamma_tf_np
beta_tf_updated  = beta_before - lr * dbeta_tf_np

gamma_np_updated = gamma_before - lr * dgamma_np
beta_np_updated  = beta_before - lr * dbeta_np

gamma_c_updated = np.ctypeslib.as_array(gamma_ptr, shape=(input_c,))
beta_c_updated  = np.ctypeslib.as_array(beta_ptr, shape=(input_c,))

# ==== Hàm in kết quả ====
def print_and_compare(name, c_output, tf_output, numpy_output):
    print(f"\n===== 🔍 So sánh {name} =====")
    print(f"➡️ TensorFlow:\n{np.round(tf_output, 5)}")
    print(f"➡️ Code C:\n{np.round(c_output, 5)}")
    print(f"➡️ NumPy chay:\n{np.round(numpy_output, 5)}")

    abs_err = np.abs(c_output - tf_output)
    rel_err = abs_err / (np.abs(tf_output) + 1e-8)
    print(f"📏 Sai số tuyệt đối (C vs TF):\n{np.round(abs_err, 5)}")
    print(f"📏 Sai số tương đối (C vs TF):\n{np.round(rel_err, 5)}")

print_and_compare("Grad Input (dx)", output_c, dx_tf_np, dx_np)
print_and_compare("Grad Gamma (dgamma)", gamma_before - gamma_c_updated, dgamma_tf_np, dgamma_np)
print_and_compare("Grad Beta (dbeta)", beta_before - beta_c_updated, dbeta_tf_np, dbeta_np)

# ==== So sánh gamma và beta sau cập nhật ====
print("\n===== 🟣 So sánh Gamma sau cập nhật =====")
print("TensorFlow Gamma:", np.round(gamma_tf_updated, 5))
print("NumPy Gamma     :", np.round(gamma_np_updated, 5))
print("Code C Gamma    :", np.round(gamma_c_updated, 5))

print("\n===== 🟣 So sánh Beta sau cập nhật =====")
print("TensorFlow Beta:", np.round(beta_tf_updated, 5))
print("NumPy Beta     :", np.round(beta_np_updated, 5))
print("Code C Beta    :", np.round(beta_c_updated, 5))

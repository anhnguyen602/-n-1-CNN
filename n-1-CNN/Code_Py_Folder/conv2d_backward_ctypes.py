import ctypes
import numpy as np
import tensorflow as tf

# ==== Load thư viện C ====
lib = ctypes.CDLL('/home/tuananh/Do an 1/-n-1-CNN/n-1-CNN/Code_C_Folder/libconv2d_backward.so')

# ==== Khai báo kiểu dữ liệu hàm C ====
lib.conv2d_backward.restype = None
lib.conv2d_backward.argtypes = [                                                       
    ctypes.POINTER(ctypes.c_float),  # grad_output
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # kernel
    ctypes.POINTER(ctypes.c_float),  # grad_input
    ctypes.POINTER(ctypes.c_float),  # grad_kernel
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float
]

# ==== Tham số ====
input_h, input_w, input_c = 3, 3, 1
kernel_h, kernel_w, output_c = 2, 2, 1
stride_h, stride_w = 1, 1
padding = 0
learning_rate = 0.01

output_h = (input_h - kernel_h + 2 * padding) // stride_h + 1
output_w = (input_w - kernel_w + 2 * padding) // stride_w + 1

# ==== Dữ liệu mẫu ====
input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
kernel_data = np.array([1, 0, 0, -1], dtype=np.float32)
grad_output_data = np.array([1, 2, 3, 4], dtype=np.float32)

grad_input = np.zeros_like(input_data)
grad_kernel = np.zeros_like(kernel_data)

# ==== Chuyển về con trỏ C ====
input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
kernel_ptr = kernel_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
grad_output_ptr = grad_output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
grad_input_ptr = grad_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
grad_kernel_ptr = grad_kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ==== Gọi hàm C ====
lib.conv2d_backward(
    grad_output_ptr,
    input_ptr,
    kernel_ptr,
    grad_input_ptr,
    grad_kernel_ptr,
    input_w, input_h, input_c,
    kernel_w, kernel_h, output_c,
    stride_w, stride_h, padding,
    ctypes.c_float(learning_rate)
)

# ==== TensorFlow raw_ops backward ====
x_tf = tf.constant(input_data.reshape(1, input_h, input_w, input_c))
w_tf = tf.constant(kernel_data.reshape(kernel_h, kernel_w, input_c, output_c))
dy_tf = tf.constant(grad_output_data.reshape(1, output_h, output_w, output_c))
x = tf.constant(np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3, 1))  # [1, 3, 3, 1]
w = tf.constant(np.array([[1, 0], [0, -1]], dtype=np.float32).reshape(2, 2, 1, 1))  # [2, 2, 1, 1]
dy = tf.constant(np.array([1, 2, 3, 4], dtype=np.float32).reshape(1, 2, 2, 1))  # [1, 2, 2, 1]
print(kernel_data)
print (w)
print(w_tf)

# Gradient theo input
grad_input_tf = tf.raw_ops.Conv2DBackpropInput(
    input_sizes=[1, input_h, input_w, input_c],
    filter=w,
    out_backprop=dy,
    strides=[1, stride_h, stride_w, 1],
    padding="VALID"
).numpy().reshape((input_c, input_h, input_w))

# Gradient theo kernel
grad_kernel_tf = tf.raw_ops.Conv2DBackpropFilter(
    input=x_tf,
    filter_sizes=[kernel_h, kernel_w, input_c, output_c],
    out_backprop=dy_tf,
    strides=[1, stride_h, stride_w, 1],
    padding="VALID"
).numpy().reshape((output_c, input_c, kernel_h, kernel_w))

# ==== Chuẩn hóa đầu ra từ C ====
grad_input_c = grad_input.reshape((input_c, input_h, input_w))
grad_kernel_c = grad_kernel.reshape((output_c, input_c, kernel_h, kernel_w))

# ==== So sánh kết quả ====
print("✅ Grad Input khớp TensorFlow:", np.allclose(grad_input_c, grad_input_tf, atol=1e-5))
print("✅ Grad Kernel khớp TensorFlow:", np.allclose(grad_kernel_c, grad_kernel_tf, atol=1e-5))

print("\n📊 Sai lệch Grad Input:")
print(np.round(grad_input_c - grad_input_tf, 5))

print("\n📊 Sai lệch Grad Kernel:")
print(np.round(grad_kernel_c - grad_kernel_tf, 5))

print("\n✅ Grad Input từ C:")
print(grad_input_c)

print("\n✅ Grad Input từ TensorFlow:")
print(grad_input_tf)

import ctypes
import numpy as np
import tensorflow as tf

# ==== Load thư viện C ====
lib = ctypes.CDLL('n-1-CNN/Code_C_Folder/libconv2d.so')

lib.conv2d.restype = None # nếu hàm không trả về gì như void thì bằng None
lib.conv2d.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # kernel
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int, #input_width
    ctypes.c_int, #input_height
    ctypes.c_int, #input_channels
    ctypes.c_int, #kernel_width
    ctypes.c_int, #kernel_height
    ctypes.c_int, #output_channels
    ctypes.c_int, #stride_width
    ctypes.c_int, #stride_height
    ctypes.c_int  #padding
]


# ==== Tham số ====
input_h, input_w, in_c = 3, 3, 1
kernel_h, kernel_w, out_c = 2, 2, 1
stride_h, stride_w, padding = 1, 1, 0
output_h = (input_h - kernel_h + 2 * padding) // stride_h + 1
output_w = (input_w - kernel_w + 2 * padding) // stride_w + 1

# ==== Dữ liệu mẫu ====
input_data = np.array([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
], dtype=np.float32)

kernel_data = np.array([
    1, 0,
    0, -1
], dtype=np.float32)

bias_data = np.array([1], dtype=np.float32)
output_data_c = np.zeros((out_c * output_h * output_w), dtype=np.float32)

# Chuyển kiểu dữ liệu từ numpy sang dạng con trỏ C (float*)
input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
kernel_ptr = kernel_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
bias_ptr = bias_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_data_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ==== Gọi hàm C ====
lib.conv2d(
    input_ptr, kernel_ptr, bias_ptr, output_ptr,
    input_w, input_h, in_c,
    kernel_w, kernel_h, out_c,
    stride_w, stride_h, padding
)

# ==== Gọi TensorFlow ====
input_tf = tf.constant(input_data.reshape(1, input_h, input_w, in_c))
kernel_tf = tf.constant(kernel_data.reshape(kernel_h, kernel_w, in_c, out_c))
bias_tf = tf.constant(bias_data)

output_tf = tf.nn.conv2d(input_tf, kernel_tf, strides=1, padding="VALID")
output_tf = tf.nn.bias_add(output_tf, bias_tf)
output_data_tf = output_tf.numpy().reshape((out_c, output_h, output_w))

# ==== So sánh kết quả ====
output_data_c_reshaped = output_data_c.reshape((out_c, output_h, output_w))

print("✅ Output from C:")
print(output_data_c_reshaped)
print("\n✅ Output from TensorFlow:")
print(output_data_tf)

if np.allclose(output_data_c_reshaped, output_data_tf, atol=1e-5):
    print("\n🎉 Kết quả TRÙNG KHỚP với TensorFlow! Hàm C của bạn ĐÚNG! ✅")
else:
    print("\n❌ Kết quả KHÔNG khớp với TensorFlow!")
    print("⚠️ Sai lệch:\n", output_data_c_reshaped - output_data_tf)

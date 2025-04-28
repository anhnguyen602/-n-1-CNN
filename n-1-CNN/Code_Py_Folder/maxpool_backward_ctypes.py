import ctypes
import numpy as np
import tensorflow as tf

# Load thư viện C
lib = ctypes.CDLL("n-1-CNN/libmyfunctions.so")  # hoặc "./maxpool.dll" trên Windows

# Cấu hình hàm
lib.maxpool_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# Tạo dữ liệu input
input_data = tf.constant([[[[1.], [3.], [2.], [1.]],
                           [[4.], [6.], [5.], [2.]],
                           [[3.], [2.], [1.], [0.]],
                           [[1.], [2.], [3.], [4.]]]], dtype=tf.float32)

ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = "VALID"

# Forward để lấy chỉ số max
pooled, argmax = tf.raw_ops.MaxPoolWithArgmax(
    input=input_data,
    ksize=ksize,
    strides=strides,
    padding=padding,
    include_batch_in_index=True,
    Targmax=tf.int32
)

grad_output = tf.ones_like(pooled)

# TensorFlow backward
grad_input_tf = tf.raw_ops.MaxPoolGrad(
    orig_input=input_data,
    orig_output=pooled,
    grad=grad_output,
    ksize=ksize,
    strides=strides,
    padding=padding
)


# Chuẩn bị dữ liệu để truyền vào C
grad_output_np = grad_output.numpy().flatten().astype(np.float32)
argmax_np = argmax.numpy().flatten().astype(np.int32)

input_shape = input_data.shape
grad_input_c = np.zeros(input_shape, dtype=np.float32).flatten()

lib.maxpool_backward(
    grad_output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    argmax_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    grad_input_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    input_shape[2], input_shape[1], input_shape[3],
    strides[2], strides[1], ksize[1]
)

# So sánh kết quả
grad_input_tf_np = grad_input_tf.numpy().flatten()

print("TensorFlow Grad Input:\n", grad_input_tf_np.reshape(input_shape))
print("C Grad Input:\n", grad_input_c.reshape(input_shape))
print("So khớp:", np.allclose(grad_input_tf_np, grad_input_c, atol=1e-6))

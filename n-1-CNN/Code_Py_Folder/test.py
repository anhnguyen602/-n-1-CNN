import tensorflow as tf
import numpy as np

# ==== Dữ liệu mẫu ====
x = tf.constant(np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3, 1))  # [1, 3, 3, 1]
w = tf.constant(np.array([[1, 0], [0, -1]], dtype=np.float32).reshape(2, 2, 1, 1))  # [2, 2, 1, 1]
dy = tf.constant(np.array([1, 2, 3, 4], dtype=np.float32).reshape(1, 2, 2, 1))  # [1, 2, 2, 1]

# ==== Gradient theo input ====
grad_input = tf.raw_ops.Conv2DBackpropInput(
    input_sizes=tf.constant([1, 3, 3, 1]),
    filter=w,
    out_backprop=dy,
    strides=[1, 1, 1, 1],
    padding="VALID"
)

# ==== Gradient theo kernel ====
grad_kernel = tf.raw_ops.Conv2DBackpropFilter(
    input=x,
    filter_sizes=tf.constant([2, 2, 1, 1]),
    out_backprop=dy,
    strides=[1, 1, 1, 1],
    padding="VALID"
)

# ==== In kết quả ====
print("🎯 Grad Input:")
print(grad_input.numpy().reshape(3, 3))

print("\n🎯 Grad Kernel:")
print(grad_kernel.numpy().reshape(2, 2))

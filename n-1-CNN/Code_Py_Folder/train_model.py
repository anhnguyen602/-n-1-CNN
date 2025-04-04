import tensorflow as tf
import numpy as np

# Load dữ liệu CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Định nghĩa mô hình KHÔNG sử dụng bias và KHÔNG dùng activation
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', use_bias=False, name='conv1', input_shape=(32, 32, 3), padding = 'same'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', use_bias=False, name='conv2', padding = 'same'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(64, activation='relu', use_bias=False, name='dense1'),
    tf.keras.layers.Dense(10, activation='softmax', use_bias=False, name='output')
])

# Compile và train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# ============================================
# ✅ Ghi input image theo HWC (height, width, channel)
# ============================================

def save_input_image_hwc(array, path):
    array = array[0]  # (1, 32, 32, 3) -> (32, 32, 3)
    H, W, C = array.shape
    with open(path, 'w') as f:
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    f.write(f"{array[h, w, c]:.6f}\n")

# Lấy ảnh đầu tiên từ tập test
sample_image = x_test[0:1]  # shape (1, 32, 32, 3)
save_input_image_hwc(sample_image, "weight/weight/input_image.txt")

# ============================================
# ✅ Ghi weights Conv2D: theo (KH, KW, IC, OC)
# ============================================

def save_weight_khwcio(weight, path):
    KH, KW, IC, OC = weight.shape
    with open(path, 'w') as f:
        for oc in range(OC):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        f.write(f"{weight[kh, kw, ic, oc]:.6f}\n")

# ============================================
# ✅ Ghi weights Dense: flatten theo (in_features, out_features)
# ============================================

def save_weight_dense_by_output(weight, path):
    """
    Ghi weight lớp Dense theo output neuron:
    → mỗi neuron output ghi toàn bộ các input weights kết nối đến nó
    """
    in_features, out_features = weight.shape
    print(weight.shape)
    with open(path, 'w') as f:
        for out_idx in range(out_features):  # Ghi theo từng output neuron
            for in_idx in range(in_features):
                # print(in_idx)
                f.write(f"{weight[in_idx, out_idx]:.25f}\n")


# Danh sách path weight
weight_paths_txt = [
    "weight/weight/conv1_weight.txt",
    "weight/weight/conv2_weight.txt",
    "weight/weight/dense1_weight.txt",
    "weight/weight/output_weight.txt"
]

# Ghi tất cả weights
weight_idx = 0
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        weight = weights[0]
        if layer.name.startswith("conv"):
            save_weight_khwcio(weight, weight_paths_txt[weight_idx])
        else:
            save_weight_dense_by_output(weight, weight_paths_txt[weight_idx])
        weight_idx += 1

# ============================================
# ✅ Ghi output của từng layer
# ============================================

def save_output_hwcf(array, path):
    # array: shape (1, H, W, C) hoặc (1, N)
    array = array[0]
    with open(path, 'w') as f:
        if array.ndim == 3:  # HWC
            W, H, C = array.shape
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        f.write(f"{array[h, w, c]:.6f}\n")
        elif array.ndim == 1:  # flatten/dense
            for val in array:
                f.write(f"{val:.6f}\n")

# Danh sách path output
output_paths_txt = [
    "weight/out/conv1_output.txt",
    "weight/out/pool1_output.txt",
    "weight/out/conv2_output.txt",
    "weight/out/pool2_output.txt",
    "weight/out/flatten_output.txt",
    "weight/out/dense1_output.txt",
    "weight/out/output_output.txt"
]

# Tạo mô hình trung gian để lấy output của từng layer
intermediate_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
outputs = intermediate_model.predict(sample_image)

# Ghi từng output
for path, output in zip(output_paths_txt, outputs):
    save_output_hwcf(output, path)

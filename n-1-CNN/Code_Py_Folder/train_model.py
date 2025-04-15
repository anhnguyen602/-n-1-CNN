import tensorflow as tf
import numpy as np

# Load dữ liệu CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Định nghĩa mô hình KHÔNG sử dụng bias và KHÔNG dùng activation trực tiếp trong Conv/Dense
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), use_bias=False, name='conv1', input_shape=(32, 32, 3), padding='same'),
    tf.keras.layers.BatchNormalization(name='batchnorm1'),
    tf.keras.layers.Activation('relu', name='relu1'),

    tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),

    tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, name='conv2', padding='same'),
    tf.keras.layers.BatchNormalization(name='batchnorm2'),
    tf.keras.layers.Activation('relu', name='relu2'),

    tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),

    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(64, use_bias=False, name='dense1'),
    tf.keras.layers.Activation('relu', name='relu3'),
    tf.keras.layers.Dense(10, activation='softmax', use_bias=False, name='output')
])

# Compile và train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.summary()

# ============================================
# ✅ Ghi input image theo HWC
# ============================================

def save_input_image_hwc(array, path):
    array = array[0]  # (1, 32, 32, 3) -> (32, 32, 3)
    H, W, C = array.shape
    with open(path, 'w') as f:
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    f.write(f"{array[h, w, c]:.6f}\n")

sample_image = x_test[0:1]
save_input_image_hwc(sample_image, "weight/weight/input_image.txt")

# ============================================
# ✅ Ghi weights Conv2D
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
# ✅ Ghi weights Dense
# ============================================

def save_weight_dense_by_output(weight, path):
    in_features, out_features = weight.shape
    with open(path, 'w') as f:
        for out_idx in range(out_features):
            for in_idx in range(in_features):
                f.write(f"{weight[in_idx, out_idx]:.25f}\n")

# Ghi weights
weight_paths_txt = [
    "weight/weight/conv1_weight.txt",
    "weight/weight/conv2_weight.txt",
    "weight/weight/dense1_weight.txt",
    "weight/weight/output_weight.txt"
]

weight_idx = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        continue
    weights = layer.get_weights()
    if weights:
        weight = weights[0]
        if layer.name.startswith("conv"):
            save_weight_khwcio(weight, weight_paths_txt[weight_idx])
        else:
            save_weight_dense_by_output(weight, weight_paths_txt[weight_idx])
        weight_idx += 1

# ============================================
# ✅ Ghi từng giá trị BatchNorm vào file riêng biệt
# ============================================

def save_array_to_file(array, path):
    with open(path, 'w') as f:
        for val in array:
            f.write(f"{val:.6f}\n")

batchnorm_layers = [
    model.get_layer('batchnorm1'),
    model.get_layer('batchnorm2')
]

batchnorm_names = ['batchnorm1', 'batchnorm2']

for layer, name in zip(batchnorm_layers, batchnorm_names):
    gamma, beta, moving_mean, moving_variance = layer.get_weights()
    save_array_to_file(gamma, f"weight/weight/{name}_gamma.txt")
    save_array_to_file(beta, f"weight/weight/{name}_beta.txt")
    save_array_to_file(moving_mean, f"weight/weight/{name}_mean.txt")
    save_array_to_file(moving_variance, f"weight/weight/{name}_variance.txt")

# ============================================
# ✅ Ghi output của các lớp chính
# ============================================

def save_output_hwcf(array, path):
    array = array[0]
    with open(path, 'w') as f:
        if array.ndim == 3:  # HWC
            W, H, C = array.shape
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        f.write(f"{array[h, w, c]:.6f}\n")
        elif array.ndim == 1:
            for val in array:
                f.write(f"{val:.6f}\n")

output_layers = [
    model.get_layer('conv1').output,
    model.get_layer('batchnorm1').output,
    model.get_layer('pool1').output,
    model.get_layer('conv2').output,
    model.get_layer('batchnorm2').output,
    model.get_layer('pool2').output,
    model.get_layer('flatten').output,
    model.get_layer('dense1').output,
    model.get_layer('output').output
]

intermediate_model = tf.keras.Model(inputs=model.input, outputs=output_layers)
outputs = intermediate_model.predict(sample_image)

output_paths_txt = [
    "weight/out/conv1_output.txt",
    "weight/out/batchnorm1_output.txt",
    "weight/out/pool1_output.txt",
    "weight/out/conv2_output.txt",
    "weight/out/batchnorm2_output.txt",
    "weight/out/pool2_output.txt",
    "weight/out/flatten_output.txt",
    "weight/out/dense1_output.txt",
    "weight/out/output_output.txt"
]

for path, output in zip(output_paths_txt, outputs):
    save_output_hwcf(output, path)

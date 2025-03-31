import tensorflow as tf
import numpy as np

# Tải bộ dữ liệu CIFAR-10 từ TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Kiểm tra kích thước của dữ liệu
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Chuyển đổi dữ liệu ảnh (0-255) thành giá trị (0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Lưu ảnh và nhãn vào các file .txt
def save_image_to_txt(image_data, filename):
    # Mỗi ảnh có kích thước 32x32x3, sẽ được chuyển thành mảng 1 chiều (3072,)
    image_data = image_data.flatten()
    np.savetxt(filename, image_data, fmt='%f')

def save_label_to_txt(label, filename):
    with open(filename, 'w') as f:
        f.write(str(label))

# Lưu 1000 ảnh đầu tiên vào các file txt
for i in range(1000):
    image_data = x_train[i]
    label = y_train[i]

    # Lưu ảnh vào file TXT
    image_filename = f"data/image_{i}.txt"
    save_image_to_txt(image_data, image_filename)

    # Lưu nhãn vào file TXT
    label_filename = f"data/label_{i}.txt"
    save_label_to_txt(label, label_filename)

print("Data processing complete!")

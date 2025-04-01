import numpy as np
import tensorflow as tf

# Hàm đọc dữ liệu từ file chứa số float với thứ tự hàng → cột → channel → filter
def read_float_file_weight(filename, shape):
    with open(filename, "r") as file:
        float_values = file.readlines()

    # Chuyển đổi từ chuỗi thành số float32
    data = np.array([float(x.strip()) for x in float_values], dtype=np.float32)

    # Định dạng lại dữ liệu theo đúng thứ tự hàng → cột → channel → filter
    H, W, C, F = shape  # File lưu theo (height → width → channel → filter)
    reshaped_data = np.zeros((H, W, C, F), dtype=np.float32)

    index = 0
    for f in range(F):  # Duyệt qua từng filter
        for c in range(C):  # Duyệt qua từng channel
            for w in range(W):  # Duyệt qua từng hàng
                for h in range(H):  # Duyệt qua từng cột
                    reshaped_data[h, w, c, f] = data[index]
                    index += 1

    return reshaped_data

# Hàm đọc dữ liệu từ file chứa số float với thứ tự hàng → cột → channel
def read_float_file(filename, shape):
    with open(filename, "r") as file:
        float_values = file.readlines()

    # Chuyển đổi từ chuỗi thành số float32
    data = np.array([float(x.strip()) for x in float_values], dtype=np.float32)

    # Định dạng lại dữ liệu theo đúng thứ tự hàng → cột → channel
    H, W, C = shape
    reshaped_data = np.zeros((H, W, C), dtype=np.float32)

    index = 0
    for h in range(H):  # Duyệt qua hàng
        for w in range(W):  # Duyệt qua cột
            for c in range(C):  # Duyệt qua channel
                reshaped_data[h, w, c] = data[index]
                index += 1

    return reshaped_data

# Hàm ghi dữ liệu ra file dưới dạng float
def write_float_file(filename, data):
    H, W, C = data.shape
    with open(filename, "w") as file:
        for c in range(C):  # Duyệt qua từng channel
            for w in range(W):  # Duyệt qua từng hàng
                for h in range(H):  # Duyệt qua từng cột
                    float_value = data[h, w, c]
                    file.write(f"{float_value:.6f}\n")  # Ghi dữ liệu dưới dạng float với 6 chữ số thập phân

# Đọc dữ liệu đầu vào
input_feature_height = 32
input_feature_width = 32
input_feature_channel = 3
weight_height = 3
weight_width = 3
weight_channel = input_feature_channel
weight_filter = 32
output_feature_height = 32
output_feature_width = 32
output_feature_channel = weight_filter

# Đường dẫn file
input_file = "data/image_0.txt"
weight_file = "weight/weights_NEW.txt"
output_file = "weight/OFM.txt"

# Đọc dữ liệu từ file input và weight (bây giờ là file float)
input_data = read_float_file(input_file, (input_feature_height, input_feature_width, input_feature_channel))
weight_data_flat = read_float_file_weight(weight_file, (weight_height, weight_width, weight_channel, weight_filter))

# Reshape lại thành (3,3,3,1) theo thứ tự hàng → cột → channel → filter
weight_data = weight_data_flat.reshape(weight_height, weight_width, weight_channel, weight_filter)

# Tạo mô hình Convolution với 1 filter
input_layer = tf.keras.layers.Input(shape=(input_feature_height, input_feature_width, input_feature_channel))
conv_layer = tf.keras.layers.Conv2D(filters=weight_filter, kernel_size=(weight_height, weight_width), padding="same", activation=None)(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=conv_layer)

# Đặt trọng số cho Conv2D layer với bias = 0
model.layers[1].set_weights([weight_data.astype(np.float32), np.zeros(weight_filter, dtype=np.float32)])

# Chạy dữ liệu qua mô hình
output_data = model.predict(input_data.reshape(1, input_feature_height, input_feature_width, input_feature_channel).astype(np.float32))
output_data = output_data.reshape(output_feature_height, output_feature_width, output_feature_channel)

# Ghi kết quả ra file dưới dạng float
write_float_file(output_file, output_data)

print(f"Kết quả đã được ghi vào {output_file}")

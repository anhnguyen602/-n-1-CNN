import matplotlib.pyplot as plt

# Đọc dữ liệu từ file
with open('weight/backward_output/accuracy_log.txt', 'r') as f:
    accuracies = [float(line.strip()) for line in f.readlines()]

# Tạo danh sách epoch (0, 1, 2, ..., n-1)
epochs = list(range(len(accuracies)))

# Vẽ đồ thị
plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracies, marker='o', linestyle='-', linewidth=2)
plt.title("Training Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.xticks(epochs)  # để hiển thị từng epoch
plt.ylim(0, 1.0)    # vì accuracy theo tỷ lệ 0-1
plt.tight_layout()
plt.show()

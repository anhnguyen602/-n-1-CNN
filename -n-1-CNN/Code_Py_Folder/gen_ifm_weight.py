import numpy as np

def save_to_hex_file(data, filename):
    with open(filename, 'w') as f:
        for val in data.flatten():
            int_val = int(val)
            f.write(f"{int_val & 0xFF:02X}\n") 

# 1. Sinh ngẫu nhiên IFM và Weight (8-bit không dấu)
np.random.seed(42)
input_feature_height = 5
input_feature_width = 5
input_feature_channel = 2
Weight_height = 3
Weight_width = 3
Weight_channel = input_feature_channel
Weight_filter = 4

IFM = np.random.randint(0, 256, size=(input_feature_height, input_feature_width, input_feature_channel), dtype=np.uint8)
Weight = np.random.randint(0, 256, size=(Weight_filter, Weight_height, Weight_width, input_feature_channel), dtype=np.uint8)

# 2. Lưu IFM và Weight vào file
save_to_hex_file(IFM, "-n-1-CNN/in-weight-out_golden/IFM.hex")
save_to_hex_file(Weight, "-n-1-CNN/in-weight-out_golden/Weight.hex")
print("✅ Đã lưu IFM và Weight vào ifm.hex, weight.hex")

# 3. Tính toàn bộ OFM (stride=1, no padding)
OH, OW = input_feature_height - Weight_height + 1, input_feature_width - Weight_width + 1  # Output height/width = 54
OFM = np.zeros((Weight_filter, OH, OW), dtype=int)  # 32 filters

print("\n🎯 BẮT ĐẦU TÍNH TOÁN TOÀN BỘ OFM (32 FILTERS)\n")

for f in range(Weight_filter):
    print(f"🔹 Filter {f}:")
    for i in range(OH):       # output height
        for j in range(OW):   # output width
            acc = 0
            is_print = (f == 1 and i == 0 and j == 0)
            if is_print:
                print(f"  ➤ Tính OFM[{f}][{i}][{j}]:")
            for kh in range(Weight_height):       # kernel height
                for kw in range(Weight_width):   # kernel width
                    for c in range(input_feature_channel):  # channels
                        ifm_val = int(IFM[i + kh, j + kw, c])
                        w_val = int(Weight[f, kh, kw, c])
                        mul = ifm_val * w_val
                        acc += mul
                        if is_print:
                            print(f"    IFM[{i+kh}][{j+kw}][{c}] = {ifm_val:3d} × "
                                  f"W[{f}][{kh}][{kw}][{c}] = {w_val:3d} → {mul:6d} "
                                  f"➕ acc = {acc}")
            OFM[f, i, j] = acc
            if is_print:
                print(f"    ➤ OFM[{f}][{i}][{j}] = {acc} (0x{acc:X})\n")
    print(f"✅ Đã xong Filter {f}.\n")

# 4. Lưu toàn bộ OFM vào file
# save_to_hex_file(OFM, "-n-1-CNN/in-weight-out_golden/OFM.hex")
print("✅ Đã lưu toàn bộ OFM vào ofm.hex")

def compare_files(file1, file2, pe_id=None):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        line_num = 1
        diff_count = 0

        while True:
            line1 = f1.readline()
            line2 = f2.readline()

            # Dừng nếu một trong hai file kết thúc
            if not line1 or not line2:
                break

            # Xử lý: xóa khoảng trắng + chuyển về chữ hoa
            clean1 = ''.join(line1.strip().split()).upper()
            clean2 = ''.join(line2.strip().split()).upper()

            if clean1 != clean2:
                if pe_id is not None:
                    print(f"❌ PE{pe_id} - Dòng {line_num} khác nhau:")
                else:
                    print(f"❌ Dòng {line_num} khác nhau:")
                print(f"    File 1: {clean1}")
                print(f"    File 2: {clean2}")
                diff_count += 1

            line_num += 1

    if diff_count == 0:
        if pe_id is not None:
            print(f"✅ PE{pe_id}: Hai file giống nhau!")
        else:
            print("✅ Hai file giống nhau!")
    else:
        print(f"⚠️ PE{pe_id}: Tổng số dòng khác nhau: {diff_count}")
    print("-" * 50)


if __name__ == "__main__":
    # Specify the file paths you want to compare
    file1 = "C:/Users/Admin/OneDrive - Hanoi University of Science and Technology/Desktop/Do an 1/-n-1-CNN/in_weight_out_C/OFM.hex"
    file2 = "C:/Users\Admin\OneDrive - Hanoi University of Science and Technology/Desktop/Do an 1/-n-1-CNN/in-weight-out_golden/OFM.hex"

    compare_files(file1, file2)  # Compare the two files

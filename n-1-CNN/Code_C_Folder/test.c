#include <stdio.h>
#include <stdlib.h>

// Hàm đọc dữ liệu từ file chứa float với thứ tự channel → hàng → cột
void read_float_file(const char *filename, float *data, int H, int W, int C) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Không thể mở file %s\n", filename);
        exit(1);
    }

    // Đọc từng giá trị float từ file và lưu vào mảng
    for (int i = 0; i < H * W * C; i++) {
        if (fscanf(file, "%f", &data[i]) != 1) {
            printf("Lỗi khi đọc file %s\n", filename);
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}

// Hàm chuyển đổi dữ liệu từ (channel, row, col) thành (row, col, channel)
void transpose_data(float *data, float *reshaped_data, int H, int W, int C) {
    int index = 0;
    for (int c = 0; c < C; c++) {   // Duyệt qua các kênh (channels)
        for (int h = 0; h < H; h++) {  // Duyệt qua hàng (rows)
            for (int w = 0; w < W; w++) {  // Duyệt qua cột (columns)
                reshaped_data[h * W * C + w * C + c] = data[index++];
            }
        }
    }
}

int main() {
    // Kích thước dữ liệu
    int H = 32;  // Số hàng (rows)
    int W = 32;  // Số cột (columns)
    int C = 3;  // Số kênh (channels)

    // Tạo mảng để lưu dữ liệu (channel, row, col)
    float *data = (float *)malloc(H * W * C * sizeof(float));

    // Đọc dữ liệu từ file vào mảng
    const char *filename = "data/image_0.txt";  // Đường dẫn đến file dữ liệu
    read_float_file(filename, data, H, W, C);

    // Tạo mảng để lưu dữ liệu đã được reshaped (row, col, channel)
    float *reshaped_data = (float *)malloc(H * W * C * sizeof(float));

    // Chuyển dữ liệu từ (channel, row, col) thành (row, col, channel)
    transpose_data(data, reshaped_data, H, W, C);

    // In dữ liệu sau khi reshaped (hàng, cột, channel)
    printf("Dữ liệu sau khi reshaped (hàng, cột, channel):\n");
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                printf("%.6f/n ", reshaped_data[h * W * C + w * C + c]);
            }
        }
    }

    // Giải phóng bộ nhớ
    free(data);
    free(reshaped_data);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void rotate_filter(double *weight, int weight_height, int weight_channels, int filter) {
    double tmp_array[50000];

    // Xoay bộ lọc (reverse các phần tử trong mỗi bộ lọc)
    for (int f = 0; f < filter * weight_channels; f++) {
        for (int j = 0; j < weight_height * weight_height; j++) {
            tmp_array[j + f * weight_height * weight_height] = weight[weight_height * weight_height - 1 - j + f * weight_height * weight_height];
        }
    }

    // Cập nhật bộ lọc với các giá trị đã xoay
    for (int i = 0; i < weight_height * weight_height * weight_channels * filter; i++) {
        weight[i] = tmp_array[i];
    }
}

void print_filter(double *weight, int weight_height, int weight_channels, int filter) {
    int idx = 0;
    for (int f = 0; f < filter; f++) {
        printf("Filter %d:\n", f + 1);
        for (int c = 0; c < weight_channels; c++) {
            printf("  Channel %d:\n", c + 1);
            for (int i = 0; i < weight_height; i++) {
                for (int j = 0; j < weight_height; j++) {
                    printf("%.2f ", weight[idx]);
                    idx++;
                }
                printf("\n");
            }
        }
    }
}

int main() {
    int weight_height = 5; // Kích thước chiều cao bộ lọc (3x3)
    int weight_channels = 3; // Chỉ có 1 kênh cho đơn giản
    int filter = 2; // Số lượng bộ lọc

    // Khởi tạo bộ lọc ngẫu nhiên
    double weight[weight_height * weight_height * weight_channels * filter];
    srand(time(0)); // Đảm bảo sinh ra các số ngẫu nhiên khác nhau mỗi lần chạy

    // Khởi tạo bộ lọc với các giá trị ngẫu nhiên
    for (int i = 0; i < weight_height * weight_height * weight_channels * filter; i++) {
        weight[i] = rand() % 10; // Các giá trị ngẫu nhiên từ 0 đến 9
    }

    printf("Bộ lọc trước khi xoay:\n");
    print_filter(weight, weight_height, weight_channels, filter);

    // Xoay bộ lọc
    rotate_filter(weight, weight_height, weight_channels, filter);

    printf("\nBộ lọc sau khi xoay:\n");
    print_filter(weight, weight_height, weight_channels, filter);

    return 0;
}

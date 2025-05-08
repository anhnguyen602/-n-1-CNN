#include"softmax.h"

void softmax(float *input, float *output, int size) {
    float sum = 0.0;
    
    // Tính tổng của exp(x_i) cho tất cả các phần tử trong mảng input
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i]); // Áp dụng exp cho từng phần tử trong input
        sum += output[i];  // Cộng dồn tổng exp(x_i)
    }

    // Chuẩn hóa kết quả để tổng bằng 1
    for (int i = 0; i < size; i++) {
        output[i] /= sum;  // Chia từng giá trị exp(x_i) cho tổng
    }
}
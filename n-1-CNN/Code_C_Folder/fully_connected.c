#include <fully_connected.h>

void fully_connected(float *input, float *weights, float *output, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j]; // Tính tổng trọng số
        }
        output[i] = relu(output[i]); // Áp dụng ReLU
    }
}
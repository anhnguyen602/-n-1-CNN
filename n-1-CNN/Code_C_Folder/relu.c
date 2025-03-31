#include<relu.h>

void relu(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;  // Áp dụng ReLU: max(0, input)
    }
}
#include <stdio.h>
void flatten_backward(
    float *grad_output, // Gradient từ lớp sau (vector WHC-flatten)
    float *grad_input,  // Gradient truyền ngược về input (HWC)
    int H, int W, int C // Kích thước input
) {
    int idx = 0;
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
             {
                int original_idx = (c * H + h) * W + w; // HWC
                grad_input[original_idx] = grad_output[idx++]; // WHC-flatten
            }
        }
    }
}
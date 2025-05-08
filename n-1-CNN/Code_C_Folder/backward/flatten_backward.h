#include <stdio.h>
void flatten_backward(
    float *grad_output, // Gradient từ lớp sau (vector WHC-flatten)
    float *grad_input,  // Gradient truyền ngược về input (HWC)
    int H, int W, int C // Kích thước input
);
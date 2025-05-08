
#include <stdio.h>
#include <stdlib.h>
void flatten_from_hwc_to_whc_flatten(
    float *input, float *output,
    int H, int W, int C
) {
    int idx = 0;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int original_idx = (c * H+ h) * W + w;  // HWC
                output[idx++] = input[original_idx];     // WHC-flatten
            }
        }
    }
}
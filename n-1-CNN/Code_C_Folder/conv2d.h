#include<stdio.h>
#include<stdint.h>
void conv2d(
    const int16_t *input,       // Pointer to input data
    const int16_t *kernel,      // Pointer to kernel weights
    const int16_t *bias,        // Pointer to bias (can be NULL)
    int16_t *output,            // Pointer to output data
    int input_width,            // Input width
    int input_height,           // Input height
    int input_channels,         // Input channels
    int kernel_width,           // Kernel width
    int kernel_height,          // Kernel height
    int output_channels,        // Number of output channels
    int stride_width,           // Stride width
    int stride_height,          // Stride height
    int padding 
);


#ifndef CONV2D_H
#define CONV2D_H
#include <stdio.h>
#include <stdlib.h>
void conv2d_backward(
    float *grad_output,      // Gradient từ lớp sau
    float *input,            // Input của Conv2D từ forward
    float *kernel,           // Kernel weights
    float *grad_input,       // Gradient truyền ngược về input
    float *grad_kernel,      // Gradient của kernel
    int input_width, int input_height, int input_channels,
    int kernel_width, int kernel_height, int output_channels,
    int stride_width, int stride_height, int padding,
    float learning_rate
) 
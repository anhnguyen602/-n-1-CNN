#ifndef MAXPOOL_H
#define MAXPOOL_H
#include <stdio.h>
#include <stdint.h>

void maxpool(
    const int16_t *input,       // Pointer to input data
    int16_t *output,            // Pointer to output data
    int input_width,            // Input width
    int input_height,           // Input height
    int input_channels,         // Input channels  
    int stride_width,           // Stride width 
    int stride_height,        // Stride height
    int pool_size
);
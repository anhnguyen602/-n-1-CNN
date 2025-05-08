#ifndef BATCHNORM_H
#define BATCHNORM_H

#include<stdio.h>
#include<stdlib.h>
void batchnorm_backward(
    const float *input,             // input xi  
    float *dL_dy,                    // grad_L_yi input cua batchNorm
    float learning_rate,
    float *gamma,
    float *beta,                     // gamma, beta
    float * mean,
    float *varience,
    int batch_size,
    float *output,                 // dL/dxi
    int input_width,
    int input_height,
    int input_channels
);
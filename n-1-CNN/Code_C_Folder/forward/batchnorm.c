#include "batchnorm.h"
#define EPSILON 0.001
void batchnorm(
    const float *input,       // Pointer to input data
    float *output,            // Pointer to output data
    float *gamma,
    float *beta,
    float *mean,
    float *variance,
    int input_width,            // Input width
    int input_height,           // Input height
    int input_channels         // Input channels
){
    for (int c = 0; c < input_channels; c++) {
        float g = gamma[c];
        float b = beta[c];
        float m = mean[c];
        float v = variance[c];
        float inv_std = 1.0f/ sqrtf(v + EPSILON);
        for (int i = 0; i < input_height * input_width; i++){
            float x = input[i + c * input_channels];
            float x_hat = (x - m) * inv_std;
            output[i + c * input_channels] = g * x_hat + b;
        }
    }
}


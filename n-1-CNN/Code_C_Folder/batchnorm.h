#include <stdlib.h>
#include <math.h>
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
);
#include "maxpool.h"
#include <stdlib.h>

void maxpool(
    const float *input,       // Pointer to input data
    float *output,            // Pointer to output data
    int input_width,            // Input width
    int input_height,           // Input height
    int input_channels,         // Input channels
    int stride_width,           // Stride width 
    int stride_height,           // Stride height
    int pool_size               // Pool size (assumed to be 2x2 for this implementation)
){
    int output_height = (input_height - pool_size) / stride_height + 1;
    int output_width = (input_width - pool_size) / stride_width + 1;
    for(int ic = 0; ic < input_channels; ic++){
        for(int oh = 0; oh < output_height;oh++){
            for(int ow = 0; ow < output_width; ow++){
                float max_value = INT16_MIN;
                for(int n = 0; n < pool_size; n++){
                    for(int m = 0; m < pool_size; m++){ 
                        int ih = oh * stride_height + n;
                        int iw = ow * stride_width + m;
                        if(ih >= 0 && ih < input_height && iw >= 0 && iw < input_width){
                            int input_idx = (ic * input_height + ih) * input_width + iw;
                            if(input[input_idx] > max_value){
                                max_value = input[input_idx];
                            }
                        }
                    }
                }                                                    
                int output_idx = (ic * output_height + oh) * output_width + ow;
                output[output_idx] = max_value;
            }
        }
    }
    
}


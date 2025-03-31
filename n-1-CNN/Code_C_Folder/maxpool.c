#include "maxpool.h"
#include <stdlib.h>

void maxpool(
    const int16_t *input,       // Pointer to input data
    int16_t *output,            // Pointer to output data
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
                int16_t max_value = INT16_MIN;
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

int main() {
    // Define the dimensions of the input and kernel
    int input_width = 4, input_height = 4, input_channels = 2;
    int pool_size = 2; // Pool size (assumed to be 2x2 for this implementation)
    int stride_width = 1, stride_height = 1, padding = 0;

    int output_height = (input_height - pool_size) / stride_height + 1;
    int output_width = (input_width - pool_size) / stride_width + 1;
    int output_size = output_width * output_height * 2 ;

    int16_t *output = (int16_t *)malloc(output_size * sizeof(int16_t));
    int16_t input[32] = {
        // Channel 1
        1, 3, 2, 1,  5, 6, 4, 2,  8, 9, 7, 3,  4, 2, 6, 5,
        10, 20, 15, 12,  25, 22, 30, 28,  35, 40, 38, 33,  45, 50, 48, 42
    };
    maxpool(input, output, input_width, input_height, input_channels, stride_width, stride_height, pool_size);

    // Print the output
    for (int i = 0; i < output_size; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

}
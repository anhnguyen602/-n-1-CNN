#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// dL/dZ (categorical_cross_entropy)
double gradient_loss_out (double predict_result, double result){
    return -result/predict_result;
}
// dL/dw = Conv2D(input, dL/dZ)
void* gradient_loss_weight(
    const double *input,                   
    const double *gradient_loss_out_array,      
    //const double *bias,                  
    double *gradient_loss_weight_array,            
    int input_width,                     
    int input_height,                       
    int input_channels,                    
    int kernel_width,                       
    int kernel_height,          
    int output_channels,        
    int stride_width,           
    int stride_height,          
    int padding         
)
{
    int padding_width = padding;
    int padding_height = padding;
    int output_height = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    int output_width = (input_width - kernel_height + 2 * padding_width) / stride_width + 1;
    
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                double value = 0; // Output value for the current pixel
                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int ih = oh * stride_height + kh - padding_height;
                            int iw = ow * stride_width + kw - padding_width;

                            // Ensure coordinates are within bounds
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = (ic * input_height + ih) * input_width + iw;
                                int weight_idx = (((oc * input_channels) + ic) * kernel_width + kh) * kernel_height + kw;
                                value += input[input_idx] * gradient_loss_out_array[weight_idx];
                            }
                        }
                    }
                }
                int output_idx = (oc * output_height + oh) * output_width + ow;
                gradient_loss_weight_array[output_idx] = value;
                
            }
        }
    }
}
// wi = wi - learning rate * dL/dW
void update (double* weight, double* gradient_loss_weight_array, double learning_rate, int n){
    for(int i = 0; i < n; i++){
        weight[i] = weight[i] - learning_rate * gradient_loss_weight_array[i];
    }
}
void rotate_filter (double *weight, int weight_height, int weight_channels, int filter){
    double tmp_array [50000];
    for (int f = 0; f < filter * weight_channels; f++){
         for (int j = 0; j < weight_height * weight_height ; j++){
            tmp_array[j + f*weight_height * weight_height] = weight[weight_height * weight_height - 1 - j + f*weight_height * weight_height];
        }
    }
    for (int i = 0; i < weight_height * weight_height * weight_channels * filter; i++){
        weight[i] = tmp_array[i];
    }
}


// dL/dI = Conv2D( padded(dL/dZ) , rotate filterK)
void* gradient_loss_input(
    const double *input,                   
    const double *gradient_loss_out_array,      
    //const double *bias,                  
    double *gradient_loss_weight_array,            
    int input_width,                     
    int input_height,                       
    int input_channels,                    
    int kernel_width,                       
    int kernel_height,          
    int output_channels,        
    int stride_width,           
    int stride_height,          
    int padding         
)
{
    int padding_width = padding;
    int padding_height = padding;
    int output_height = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    int output_width = (input_width - kernel_height + 2 * padding_width) / stride_width + 1;
    
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                double value = 0; // Output value for the current pixel
                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int ih = oh * stride_height + kh - padding_height;
                            int iw = ow * stride_width + kw - padding_width;

                            // Ensure coordinates are within bounds
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = (ic * input_height + ih) * input_width + iw;
                                int weight_idx = (((oc * input_channels) + ic) * kernel_width + kh) * kernel_height + kw;
                                value += input[input_idx] * gradient_loss_out_array[weight_idx];
                            }
                        }
                    }
                }
                int output_idx = (oc * output_height + oh) * output_width + ow;
                gradient_loss_weight_array[output_idx] = value;
                
            }
        }
    }
}
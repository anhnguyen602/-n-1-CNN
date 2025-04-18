#include <stdio.h>
#include <stdlib.h>
void rotate_filter (float *weight_in, float *weight_out, int weight_height, int weight_channels, int filter){
    float tmp_array [50000];
    for (int f = 0; f < filter * weight_channels; f++){
         for (int j = 0; j < weight_height * weight_height ; j++){
            tmp_array[j + f*weight_height * weight_height] = weight_in[weight_height * weight_height - 1 - j + f*weight_height * weight_height];
        }
    }
    for (int i = 0; i < weight_height * weight_height * weight_channels * filter; i++){
        weight_out[i] = tmp_array[i];
    }
}
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
) {
    int output_height = (input_height - kernel_height + 2 * padding) / stride_height + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride_width + 1;
    float *kernel1 = (float*)malloc(kernel_width*kernel_height*output_channels*input_channels * sizeof(float));
    // rotate_filter(kernel, kernel1, kernel_height, input_channels, output_channels);
    // Khởi tạo grad_input và grad_kernel bằng 0
    for (int i = 0; i < input_height * input_width * input_channels; i++) {
        grad_input[i] = 0.0f;
    }
    for (int i = 0; i < kernel_width * kernel_height * input_channels * output_channels; i++) {
        grad_kernel[i] = 0.0f;
    }

    // Tính gradient
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                int output_idx = (oc * output_height + oh) * output_width + ow;
                float grad = grad_output[output_idx];

                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int ih = oh * stride_height + kh - padding;
                            int iw = ow * stride_width + kw - padding;
                            int weight_idx = (((oc * input_channels) + ic) * kernel_height + kh) * kernel_width + kw;

                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = (ic * input_height + ih) * input_width + iw;
                                // Gradient cho kernel
                                grad_kernel[weight_idx] += grad * input[input_idx];
                                // Gradient cho input
                                grad_input[input_idx] += grad * kernel[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Cập nhật kernel
    for (int i = 0; i < kernel_width * kernel_height * input_channels * output_channels; i++) {
        kernel[i] -= learning_rate * grad_kernel[i];
    }
}
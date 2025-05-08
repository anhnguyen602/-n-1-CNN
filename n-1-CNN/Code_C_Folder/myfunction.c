#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
// #include "conv2d.h"
// #include "relu.h"
// #include "fully_connected.h"
// #include "maxpool.h"
// #include "softmax.h"

//biến đổi từ whc -> cwh
//flatten forward
void flatten_from_hwc_to_whc_flatten(
    float *input, float *output,
    int H, int W, int C
) {
    int idx = 0;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int original_idx = (c * H+ h) * W + w;  // HWC
                output[idx++] = input[original_idx];     // WHC-flatten
            }
        }
    }
}
void flatten_backward(
    float *grad_output, // Gradient từ lớp sau (vector WHC-flatten)
    float *grad_input,  // Gradient truyền ngược về input (HWC)
    int H, int W, int C // Kích thước input
) {
    int idx = 0;
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
             {
                int original_idx = (c * H + h) * W + w; // HWC
                grad_input[original_idx] = grad_output[idx++]; // WHC-flatten
            }
        }
    }
}
void save_float_array_to_txt_file(const char *filename, float *array, int size) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("⚠️ Không thể mở file %s để ghi!\n", filename);
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(f, "%.6f\n", array[i]);  // Ghi 6 chữ số thập phân
    }

    fclose(f);
}

int read_label_from_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening label file %s\n", filename);
        exit(1);
    }

    int label;
    // Đọc số trong dấu ngoặc vuông, ví dụ: [6]
    fscanf(file, "[%d]", &label);  // Đọc label trong dấu ngoặc vuông
    fclose(file);
    return label;
}


int get_max_label(float *output, int size) {
    int max_label = 0;
    float max_value = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_label = i;
        }
    }
    return max_label;
}
//softmax forward
void softmax(float *input, float *output, int size) {
    float sum = 0.0;
    
    // Tính tổng của exp(x_i) cho tất cả các phần tử trong mảng input
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i]); // Áp dụng exp cho từng phần tử trong input
        sum += output[i];  // Cộng dồn tổng exp(x_i)
    }

    // Chuẩn hóa kết quả để tổng bằng 1
    for (int i = 0; i < size; i++) {
        output[i] /= sum;  // Chia từng giá trị exp(x_i) cho tổng
    }
}
//softmax backward
void softmax_cross_entropy_derivative(float *y_pred, float *y_true, float *grad, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] = y_pred[i] - y_true[i]; // Gradient của loss đối với output của Softmax
    }
}

//relu forward
void relu(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;  // Áp dụng ReLU: max(0, input)
    }
}
//relu backward
void relu_backward(
    float *grad_output, // Gradient từ lớp sau
    float *input,       // Input của ReLU từ forward
    float *grad_input,  // Gradient truyền ngược về input
    int size            // Kích thước mảng
) {
    for (int i = 0; i < size; i++) {
        // Nếu input[i] > 0, truyền gradient ngược lại, ngược lại đặt bằng 0
        grad_input[i] = (input[i] > 0) ? grad_output[i] : 0.0f;
    }
}

//maxpool forward
void maxpool(
    const float *input,       // Pointer to input data
    float *output,            // Pointer to output data
    int *max_indices,       // (thêm mới)Pointer to store max indices (optional) 
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
                float max_value = -999.0;
                int max_idx = 0;
                for(int n = 0; n < pool_size; n++){
                    for(int m = 0; m < pool_size; m++){ 
                        int ih = oh * stride_height + n;
                        int iw = ow * stride_width + m;
                        
                            int input_idx = (ic * input_height + ih) * input_width + iw;
                            if(input[input_idx] > max_value){
                                max_value = input[input_idx];
                                max_idx = input_idx;//thêm mới
                            }
                        
                    }
                }                                                    
                int output_idx = (ic * output_height + oh) * output_width + ow;
                output[output_idx] = max_value;
                max_indices[output_idx] = max_idx;  //thêm mới
            }
        }
    }
    
}
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
//maxpool backward
void maxpool_backward(
    float *grad_output,      // Gradient từ lớp sau
    int *max_indices,        // Chỉ số max từ forward
    float *grad_input,       // Gradient truyền ngược về input
    int input_width, int input_height, int input_channels,
    int stride_width, int stride_height, int pool_size
) {
    int output_height = (input_height - pool_size) / stride_height + 1;
    int output_width = (input_width - pool_size) / stride_width + 1;

    // Khởi tạo grad_input bằng 0
    for (int i = 0; i < input_height * input_width * input_channels; i++) {
        grad_input[i] = 0.0f;
    }

    // Truyền gradient về các vị trí max
    for (int ic = 0; ic < input_channels; ic++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                int output_idx = (ic * output_height + oh) * output_width + ow;
                int max_idx = max_indices[output_idx];
                grad_input[max_idx] = grad_output[output_idx];
            }
        }
    }
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// void maxpool_backward(float* input, float* dy, float* grad,
//                      int batch, int in_height, int in_width, int channels,
//                      int out_height, int out_width,
//                      int pool_height, int pool_width,
//                      int stride_height, int stride_width) {
//     memset(grad, 0, batch * in_height * in_width * channels * sizeof(float));

//     for (int b = 0; b < batch; b++) {
//         for (int oh = 0; oh < out_height; oh++) {
//             for (int ow = 0; ow < out_width; ow++) {
//                 for (int c = 0; c < channels; c++) {
//                     int max_i = -1;
//                     float max_val = -1e9;
//                     printf("Processing b=%d, oh=%d, ow=%d, c=%d\n", b, oh, ow, c);
//                     for (int ph = 0; ph < pool_height; ph++) {
//                         for (int pw = 0; pw < pool_width; pw++) {
//                             int ih = oh * stride_height + ph;
//                             int iw = ow * stride_width + pw;
//                             if (ih < in_height && iw < in_width) {
//                                 int idx = ((b * in_height + ih) * in_width + iw) * channels + c;
//                                 printf("  ih=%d, iw=%d, idx=%d, input[idx]=%f\n", ih, iw, idx, input[idx]);
//                                 if (input[idx] > max_val) {
//                                     max_val = input[idx];
//                                     max_i = idx;
//                                     printf("  New max: max_val=%f, max_i=%d\n", max_val, max_i);
//                                 }
//                             } else {
//                                 printf("  Skipping ih=%d, iw=%d (out of bounds)\n", ih, iw);
//                             }
//                         }
//                     }
//                     if (max_i >= 0) {
//                         int out_idx = ((b * out_height + oh) * out_width + ow) * channels + c;
//                         printf("Assigning dy[%d]=%f to grad[%d]\n", out_idx, dy[out_idx], max_i);
//                         grad[max_i] = dy[out_idx];
//                     } else {
//                         printf("No max found for b=%d, oh=%d, ow=%d, c=%d\n", b, oh, ow, c);
//                     }
//                 }
//             }
//         }
//     }
// }


//fully_connected forward
void fully_connected(float *input, float *weights, float *output, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j]; // Tính tổng trọng số
        }
        //output[i] = relu(output[i]); // Áp dụng ReLU
    }
}

//fully_connected backward
void fully_connected_backward(
    float *grad_output,      // Gradient từ lớp sau (đối với output của FC)
    float *input,            // Input của FC từ forward
    float *weights,          // Weights của FC
    float *grad_input,       // Gradient truyền ngược về input
    float *grad_weights,     // Gradient của weights
    int input_size,          // Kích thước input
    int output_size,         // Kích thước output
    float learning_rate      // Tốc độ học
) {
    // Khởi tạo grad_input bằng 0
    for (int j = 0; j < input_size; j++) {
        grad_input[j] = 0.0f;
    }

    // Tính gradient cho weights và grad_input
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            // Gradient cho weights: dW[j,i] = grad_output[i] * input[j]
            int weight_idx = i * input_size + j;
            grad_weights[weight_idx] = grad_output[i] * input[j];
            // Gradient cho input: grad_input[j] += W[j,i] * grad_output[i]
            grad_input[j] += weights[weight_idx] * grad_output[i];
        }
    }

    // Cập nhật weights
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}
//conv2d forward
void conv2d(
    const float *input,       // Pointer to input data
    const float *kernel,      // Pointer to kernel weights
    const float *bias,        // Pointer to bias (can be NULL)
    float *output,            // Pointer to output data
    int input_width,            // Input width
    int input_height,           // Input height
    int input_channels,         // Input channels
    int kernel_width,           // Kernel width
    int kernel_height,          // Kernel height
    int output_channels,        // Number of output channels
    int stride_width,           // Stride width
    int stride_height,          // Stride height
    int padding         // Padding type
)
{
    int padding_width = padding;
    int padding_height = padding;
    int output_height = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    int output_width = (input_width - kernel_height + 2 * padding_width) / stride_width + 1;
    
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                float value = 0; // Output value for the current pixel
                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int ih = oh * stride_height + kh - padding_height;
                            int iw = ow * stride_width + kw - padding_width;

                            // Ensure coordinates are within bounds
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = (ic * input_height + ih) * input_width + iw;
                                int weight_idx = (((oc * input_channels) + ic) * kernel_width + kh) * kernel_height + kw;
                                value += input[input_idx] * kernel[weight_idx];
                            }
                        }
                    }
                }
                int output_idx = (oc * output_height + oh) * output_width + ow;
                if (bias != NULL) {
                    output[output_idx] = value + bias[oc];
                } else {
                    output[output_idx] = value;
                }
            }
        }
    }
}
void read_from_file(const char *filename, float *array, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &array[i]);
    }

    fclose(file);
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
    rotate_filter(kernel, kernel1, kernel_height, input_channels, output_channels);
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
                                grad_input[input_idx] += grad * kernel1[weight_idx];
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
){
    #define EPSILON 0.001
    int m = input_height * input_width * batch_size;
   // float dL_dxi[input_channels*input_height*input_width*batch_size] = {0};
    float dL_dgamma[input_channels] ;
    float dL_dBeta[input_channels];
    float dL_dhat_xi[input_channels*input_height*input_width*batch_size] ;
    float dL_dsigma_B_binh[input_channels]  ;
    float dL_d_mu_B[input_channels];
   // tinh dL_dgamma, dL_dBeta
    for (int i = 0; i< input_channels; i++){
        int index_of_mu_thu_i = i*input_height*input_width;         // xac dinh vi tri bat dau cua kenh thu i cua moi batch
        float sum_dL_dy_gamma = 0.0f;
        float sum_dL_dy_Beta = 0.0f;
        float sum_sigma_B_binh = 0.0f;
        float sum_dL_hat_xi = 0.0f;
        float d_sigma_B_binh_d_muB = 0.0f;
        for (int j = 0; j< batch_size;j++){
            int index_start_of_channels = j *input_height*input_width*input_channels + index_of_mu_thu_i;          // xac dinh vi tri cua kennh thu i cua batch thu j
            for (int k = 0; k <  input_height*input_width; k++){
                float hat_xi = (input[k + index_start_of_channels] - mean[i]) / (sqrt(varience[i]*varience[i] + EPSILON));
                sum_dL_dy_gamma += dL_dy[k+index_start_of_channels] * hat_xi;
                sum_dL_dy_Beta += dL_dy[k + index_start_of_channels];
                dL_dhat_xi[k + index_start_of_channels] = dL_dy[k+index_start_of_channels]*gamma[i];
                sum_sigma_B_binh += dL_dhat_xi[k + index_start_of_channels] *(input[index_start_of_channels + k]- mean[i]); 
                sum_dL_hat_xi += dL_dhat_xi[k + index_start_of_channels];
                d_sigma_B_binh_d_muB += (1/(m)) * (-2)* (input[k + index_start_of_channels] - mean[i]);
            }
        }
        dL_dgamma[i] = sum_dL_dy_gamma;
        dL_dBeta[i] = sum_dL_dy_Beta;
        dL_dsigma_B_binh[i] = sum_sigma_B_binh* (-0.5)* (1/sqrt((varience[i] *varience[i] + EPSILON)*(varience[i] *varience[i] + EPSILON)*(varience[i] *varience[i] + EPSILON)));
        dL_d_mu_B[i] = sum_dL_hat_xi* (-1/sqrt(varience[i] * varience[i] + EPSILON ) ) + dL_dsigma_B_binh[i] * d_sigma_B_binh_d_muB;
        gamma[i] = gamma[i] - learning_rate*dL_dgamma[i];
        beta[i] = beta[i]- learning_rate*dL_dBeta[i];
 
    }   
    // inh dL_dxi(output)
    for (int i = 0; i< input_channels; i++){
        int index_of_mu_thu_i = i*input_height*input_width;         // xac dinh vi tri bat dau cua kenh thu i cua moi batch
        for (int j = 0; j< batch_size;j++){
            int index_start_of_chanels = j *input_height*input_width*input_channels + index_of_mu_thu_i;          // xac dinh vi tri cua kennh thu i cua batch thu j
            for (int k = 0; k< input_height*input_width; k++){
               output[k + index_start_of_chanels] = dL_dhat_xi[k + index_start_of_chanels] * 1/(sqrt(varience[i] * varience[i] + EPSILON)) + dL_dsigma_B_binh[i] * 1/m * 2*(input[k + index_start_of_chanels]) + dL_d_mu_B[i] * 1/m;
            }
        }
    }

}

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
// #include "conv2d.h"
// #include "relu.h"
// #include "fully_connected.h"
// #include "maxpool.h"
// #include "softmax.h"
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
            float x = input[i + c *input_height * input_width ];
            float x_hat = (x - m) * inv_std;
            output[i + c * input_height * input_width] = g * x_hat + b;
        }
    }
}
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
void relu(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;  // Áp dụng ReLU: max(0, input)
    }
}
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
                float max_value = -999.0;
                for(int n = 0; n < pool_size; n++){
                    for(int m = 0; m < pool_size; m++){ 
                        int ih = oh * stride_height + n;
                        int iw = ow * stride_width + m;
                        
                            int input_idx = (ic * input_height + ih) * input_width + iw;
                            if(input[input_idx] > max_value){
                                max_value = input[input_idx];
                            }
                        
                    }
                }                                                    
                int output_idx = (ic * output_height + oh) * output_width + ow;
                output[output_idx] = max_value;
            }
        }
    }
    
}
void fully_connected(float *input, float *weights, float *output, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j]; // Tính tổng trọng số
        }
        //output[i] = relu(output[i]); // Áp dụng ReLU
    }
}
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



int main() {
    // ⚙️ Kích thước đầu vào & tham số model
    const int input_width = 32, input_height = 32, input_channels = 3;
    const int output_size = 10;
    const int kernel_size = 3, padding = 1;
    const int conv1_channels = 32;
    const int conv2_channels = 64;
    const int dense1_size = 64;
    const int flatten_size = 8 * 8 * conv2_channels;  // sau 2 lần pooling

    // ⚙️ Khai báo trọng số (đã được xuất từ Python)
    float kernel1[kernel_size * kernel_size * input_channels * conv1_channels];
    float kernel2[kernel_size * kernel_size * conv1_channels * conv2_channels];
    float weight_dense1[flatten_size * dense1_size];
    float weight_output[dense1_size * output_size];

    float bn1_gamma[conv1_channels], bn1_beta[conv1_channels], bn1_mean[conv1_channels], bn1_variance[conv1_channels];
    float bn2_gamma[conv2_channels], bn2_beta[conv2_channels], bn2_mean[conv2_channels], bn2_variance[conv2_channels];


    // ⚙️ Bias = NULL (không dùng)
    float* bias_null = NULL;

    // ⚙️ Load weights từ file
    read_from_file("weight/weight/conv1_weight.txt", kernel1, sizeof(kernel1) / sizeof(float));
    read_from_file("weight/weight/conv2_weight.txt", kernel2, sizeof(kernel2) / sizeof(float));
    read_from_file("weight/weight/dense1_weight.txt", weight_dense1, sizeof(weight_dense1) / sizeof(float));
    read_from_file("weight/weight/output_weight.txt", weight_output, sizeof(weight_output) / sizeof(float));

        // BatchNorm 1
    read_from_file("weight/weight/batchnorm1_gamma.txt", bn1_gamma, conv1_channels);
    read_from_file("weight/weight/batchnorm1_beta.txt", bn1_beta, conv1_channels);
    read_from_file("weight/weight/batchnorm1_mean.txt", bn1_mean, conv1_channels);
    read_from_file("weight/weight/batchnorm1_variance.txt", bn1_variance, conv1_channels);

    // BatchNorm 2
    read_from_file("weight/weight/batchnorm2_gamma.txt", bn2_gamma, conv2_channels);
    read_from_file("weight/weight/batchnorm2_beta.txt", bn2_beta, conv2_channels);
    read_from_file("weight/weight/batchnorm2_mean.txt", bn2_mean, conv2_channels);
    read_from_file("weight/weight/batchnorm2_variance.txt", bn2_variance, conv2_channels);


    // ⚙️ Buffer cho từng layer
    float input[32 * 32 * 3];
    float output_conv1[32 * 32 * conv1_channels];
    float output_pool1[16 * 16 * conv1_channels];
    float output_conv2[16 * 16 * conv2_channels];
    float output_pool2[8 * 8 * conv2_channels];
    float flatten[8*8*conv2_channels];
    float output_fc1[dense1_size];
    float output_fc2[output_size];
    float output_bn1[32 * 32 * conv1_channels];
    float output_bn2[16 * 16 * conv2_channels];

    float output_bn1_raw[32 * 32 * conv1_channels];
    float output_bn2_raw[16 * 16 * conv2_channels];


    int correct_predictions = 0;

    for (int i = 0; i < 1000; i++) {
        // 🧠 Đọc input & label
        char input_file[50], label_file[50];
        snprintf(input_file, sizeof(input_file), "data/image_%d.txt", i);
        snprintf(label_file, sizeof(label_file), "data/label_%d.txt", i);
        read_from_file(input_file, input, 32 * 32 * 3);
        // for (int i = 0; i < 10; i++){
        //     printf("%f\n", input[i]);
        // }   
        int true_label = read_label_from_file(label_file);

        // 🧪 Conv1 → ReLU → Pool
        conv2d(input, kernel1, bias_null, output_conv1,
            input_width, input_height, input_channels,
            kernel_size, kernel_size, conv1_channels,
            1, 1, padding);

    batchnorm(output_conv1, output_bn1_raw, bn1_gamma, bn1_beta, bn1_mean, bn1_variance,
            32, 32, conv1_channels);

    relu(output_bn1_raw, output_bn1, 32 * 32 * conv1_channels);
    maxpool(output_bn1, output_pool1, 32, 32, conv1_channels, 2, 2, 2);


    // 🧪 Conv2 → BatchNorm → ReLU → Pool
    conv2d(output_pool1, kernel2, bias_null, output_conv2,
        16, 16, conv1_channels,
        kernel_size, kernel_size, conv2_channels,
        1, 1, padding);

    batchnorm(output_conv2, output_bn2_raw, bn2_gamma, bn2_beta, bn2_mean, bn2_variance,
        16, 16, conv2_channels);

    relu(output_bn2_raw, output_bn2, 16 * 16 * conv2_channels);
    maxpool(output_bn2, output_pool2, 16, 16, conv2_channels, 2, 2, 2);


    // 🧠 Dense1 → ReLU
    flatten_from_hwc_to_whc_flatten(output_pool2 , flatten, 8, 8, 64);

    fully_connected(flatten, weight_dense1, output_fc1, flatten_size, dense1_size);
    relu(output_fc1, output_fc1, dense1_size);

    // 🧠 Output → Softmax
    fully_connected(output_fc1, weight_output, output_fc2, dense1_size, output_size);
    softmax(output_fc2, output_fc2, output_size);

        // 🎯 Dự đoán và so sánh
        int predicted_label = get_max_label(output_fc2, output_size);
        if (predicted_label == true_label)
            correct_predictions++;

        printf("Image %d - Predict: %d, True: %d\n", i, predicted_label, true_label);
    }

    printf("\n✅ Accuracy: %.2f%% (%d / 1000 correct)\n",
           (float)correct_predictions / 100.0, correct_predictions);

    return 0;
}


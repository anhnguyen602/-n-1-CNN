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
void init_random_array(float *array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // từ -1.0 đến 1.0
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


int main() {
    printf("⚙️ Bắt đầu chạy model...\n");
    // ⚙️ Kích thước đầu vào & tham số model
    const int input_width = 32, input_height = 32, input_channels = 3;
    const int output_size = 10;
    const int kernel_size = 3, padding = 1;
    const int conv1_channels = 32;
    const int conv2_channels = 64;
    const int dense1_size = 64;
    const int flatten_size = 8 * 8 * conv2_channels;
    float learning_rate = 0.001f;

    // ⚙️ Khai báo trọng số
    float kernel1[kernel_size * kernel_size * input_channels * conv1_channels];
    float kernel2[kernel_size * kernel_size * conv1_channels * conv2_channels];
    float weight_dense1[flatten_size * dense1_size];
    float weight_output[dense1_size * output_size];
    float* bias_null = NULL;

    // ⚙️ Gradient buffers
    float grad_kernel1[kernel_size * kernel_size * input_channels * conv1_channels];
    float grad_kernel2[kernel_size * kernel_size * conv1_channels * conv2_channels];
    float grad_weight_dense1[flatten_size * dense1_size];
    float grad_weight_output[dense1_size * output_size];

    // ⚙️ Load weights
    read_from_file("weight/weight/conv1_weight.txt", kernel1, sizeof(kernel1) / sizeof(float));
    read_from_file("weight/weight/conv2_weight.txt", kernel2, sizeof(kernel2) / sizeof(float));
    read_from_file("weight/weight/dense1_weight.txt", weight_dense1, sizeof(weight_dense1) / sizeof(float));
    read_from_file("weight/weight/output_weight.txt", weight_output, sizeof(weight_output) / sizeof(float));

    // init_random_array(kernel1, sizeof(kernel1) / sizeof(float));
    // init_random_array(kernel2, sizeof(kernel2) / sizeof(float));
    // init_random_array(weight_dense1, sizeof(weight_dense1) / sizeof(float));
    // init_random_array(weight_output, sizeof(weight_output) / sizeof(float));

    FILE *acc_file = fopen("weight/backward_output/accuracy_log.txt", "w");
    if (!acc_file) {
        printf("⚠️ Không thể mở file accuracy_log.txt để ghi!\n");
        return 1;
    }
    // ⚙️ Buffers
    float input[32 * 32 * 3];
    float output_conv1[32 * 32 * conv1_channels];
    float output_pool1[16 * 16 * conv1_channels];
    float output_conv2[16 * 16 * conv2_channels];
    float output_pool2[8 * 8 * conv2_channels];
    float flatten[8 * 8 * conv2_channels];
    float output_fc1[dense1_size];
    float output_fc2[output_size];
    float output [output_size];
    int max_indices_pool1[16 * 16 * conv1_channels];
    int max_indices_pool2[8 * 8 * conv2_channels];

    // ⚙️ Gradient buffers cho backward
    float grad_output_fc2[output_size];
    float grad_output_fc1[dense1_size];
    float grad_flatten[8 * 8 * conv2_channels];
    float grad_output_pool2[8 * 8 * conv2_channels];
    float grad_output_conv2[16 * 16 * conv2_channels];
    float grad_output_pool1[16 * 16 * conv1_channels];
    float grad_output_conv1[32 * 32 * conv1_channels];
    float grad_input[32 * 32 * 3];

    int correct_predictions = 0;

    for (int epoch = 0; epoch < 10; epoch++) { // Thêm vòng lặp epoch
        correct_predictions = 0;
        for (int i = 0; i < 10000; i++) {
            // 🧠 Đọc input & label
            char input_file[50], label_file[50];
            snprintf(input_file, sizeof(input_file), "data/image_%d.txt", i);
            snprintf(label_file, sizeof(label_file), "data/label_%d.txt", i);
            read_from_file(input_file, input, 32 * 32 * 3);
            int true_label = read_label_from_file(label_file);
            printf("done\n");
            // 🧪 Forward pass
            conv2d(input, kernel1, bias_null, output_conv1,
                   input_width, input_height, input_channels,
                   kernel_size, kernel_size, conv1_channels,
                   1, 1, padding);
            relu(output_conv1, output_conv1, 32 * 32 * conv1_channels);
            maxpool(output_conv1, output_pool1, max_indices_pool1,
                    32, 32, conv1_channels, 2, 2, 2);

            conv2d(output_pool1, kernel2, bias_null, output_conv2,
                   16, 16, conv1_channels,
                   kernel_size, kernel_size, conv2_channels,
                   1, 1, padding);
            relu(output_conv2, output_conv2, 16 * 16 * conv2_channels);
            maxpool(output_conv2, output_pool2, max_indices_pool2,
                    16, 16, conv2_channels, 2, 2, 2);

            flatten_from_hwc_to_whc_flatten(output_pool2, flatten, 8, 8, conv2_channels);
            fully_connected(flatten, weight_dense1, output_fc1, flatten_size, dense1_size);
            relu(output_fc1, output_fc1, dense1_size);
            fully_connected(output_fc1, weight_output, output_fc2, dense1_size, output_size);
            softmax(output_fc2, output, output_size);

            // 🎯 Dự đoán
            int predicted_label = get_max_label(output, output_size);
            if (predicted_label == true_label)
                correct_predictions++;

            // 🧪 Backward pass
            float y_true[output_size];  // Khai báo mảng trước
            memset(y_true, 0, sizeof(y_true)); // Gán tất cả phần tử về 0
            
            y_true[true_label] = 1.0f;
            softmax_cross_entropy_derivative(output, y_true, grad_output_fc2, output_size);
            fully_connected_backward(grad_output_fc2, output_fc1, weight_output, grad_output_fc1,
                                    grad_weight_output, dense1_size, output_size, learning_rate);
            relu_backward(grad_output_fc1, output_fc1, grad_output_fc1, dense1_size);
            fully_connected_backward(grad_output_fc1, flatten, weight_dense1, grad_flatten,
                                    grad_weight_dense1, flatten_size, dense1_size, learning_rate);
            flatten_backward(grad_flatten, grad_output_pool2, 8, 8, conv2_channels);
            maxpool_backward(grad_output_pool2, max_indices_pool2, grad_output_conv2,
                             16, 16, conv2_channels, 2, 2, 2);
            relu_backward(grad_output_conv2, output_conv2, grad_output_conv2, 16 * 16 * conv2_channels);
            
            conv2d_backward(grad_output_conv2, output_pool1, kernel2, grad_output_pool1, grad_kernel2,
                            16, 16, conv1_channels, kernel_size, kernel_size, conv2_channels,
                            1, 1, padding, learning_rate);
            maxpool_backward(grad_output_pool1, max_indices_pool1, grad_output_conv1,
                             32, 32, conv1_channels, 2, 2, 2);
            relu_backward(grad_output_conv1, output_conv1, grad_output_conv1, 32 * 32 * conv1_channels);
            conv2d_backward(grad_output_conv1, input, kernel1, grad_input, grad_kernel1,
                            input_width, input_height, input_channels, kernel_size, kernel_size,
                            conv1_channels, 1, 1, padding, learning_rate);

           printf("Epoch: %d Image %d - Predict: %d, True: %d\n",epoch, i, predicted_label, true_label);
        }
        printf("Epoch %d - Accuracy: %.2f%% (%d / 100 correct)\n",
               epoch, (float)correct_predictions / 10.0, correct_predictions);
        fprintf(acc_file, "%.2f\n", (float)correct_predictions / 10.0);

        
    }
        // ✅ GHI OUTPUT CỦA CÁC LỚP (FORWARD)
    // save_float_array_to_txt_file("weight/forward_output/conv1.txt", output_conv1, 32 * 32 * conv1_channels);
    // save_float_array_to_txt_file("weight/forward_output/pool1.txt", output_pool1, 16 * 16 * conv1_channels);
    // save_float_array_to_txt_file("weight/forward_output/conv2.txt", output_conv2, 16 * 16 * conv2_channels);
    // save_float_array_to_txt_file("weight/forward_output/pool2.txt", output_pool2, 8 * 8 * conv2_channels);
    // save_float_array_to_txt_file("weight/forward_output/flatten.txt", flatten, flatten_size);
    // save_float_array_to_txt_file("weight/forward_output/dense1.txt", output_fc1, dense1_size);
    // save_float_array_to_txt_file("weight/forward_output/output.txt", output_fc2, output_size);

    // ✅ GHI GRADIENT CỦA CÁC LỚP (BACKWARD)
    // ✅ GHI TRỌNG SỐ ĐÃ CẬP NHẬT (SAU BACKPROP)
    save_float_array_to_txt_file("weight/backward_output/kernel1_updated.txt", kernel1, kernel_size * kernel_size * input_channels * conv1_channels);
    save_float_array_to_txt_file("weight/backward_output/kernel2_updated.txt", kernel2, kernel_size * kernel_size * conv1_channels * conv2_channels);
    save_float_array_to_txt_file("weight/backward_output/dense1_weights_updated.txt", weight_dense1, flatten_size * dense1_size);
    save_float_array_to_txt_file("weight/backward_output/output_weights_updated.txt", weight_output, dense1_size * output_size);
    fclose(acc_file);

    return 0;
}

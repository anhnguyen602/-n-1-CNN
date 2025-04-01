#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
// #include "conv2d.h"
// #include "relu.h"
// #include "fully_connected.h"
// #include "maxpool.h"
// #include "softmax.h"
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
        // output[i] = relu(output[i]); // Áp dụng ReLU
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
void transpose_data(float *data, float *reshaped_data, int H, int W, int C) {
    int index = 0;
    for (int c = 0; c < C; c++) {   // Duyệt qua các kênh (channels)
        for (int h = 0; h < H; h++) {  // Duyệt qua hàng (rows)
            for (int w = 0; w < W; w++) {  // Duyệt qua cột (columns)
                reshaped_data[h * W * C + w * C + c] = data[index++];
            }
        }
    }
}

int main() {
    // Các tham số kích thước đầu vào và số lượng lớp Fully Connected
    int input_width = 32, input_height = 32, input_channels = 3;
    int output_size = 10;  // Ví dụ lớp fully connected có 10 đầu ra (phân loại 10 lớp)

    // Các tham số cho kích thước các lớp Conv2D
    int kernel_size = 3;   // Kích thước của kernel (3x3)
    int channels_conv1 = 32, channels_conv2 = 32, channels_conv3 = 64, channels_conv4 = 64;
    int padding = 1;

    // Giả định dữ liệu đầu vào (sử dụng giá trị ngẫu nhiên cho đơn giản)

    // read_from_file("data/image_1.txt", input, 32 * 32 * 3);
    // // Khai báo trọng số và bias cho các lớp Conv2D
    float kernel1[kernel_size * kernel_size * input_channels * channels_conv1];
    float kernel2[kernel_size * kernel_size * channels_conv1 * channels_conv2];
    float kernel3[kernel_size * kernel_size * channels_conv2 * channels_conv3];
    float kernel4[kernel_size * kernel_size * channels_conv3 * channels_conv4];
    float weights_fc[8 * 8 * 64 * output_size]; // Trọng số lớp fully connected
    float bias1[32] = {0};  // Khởi tạo bias = 0
    float bias2[32] = {0};  // Khởi tạo bias = 0
    float bias3[64] = {0};  // Khởi tạo bias = 0
    float bias4[64] = {0};  // Khởi tạo bias = 0

    read_from_file("weight/weights_NEW_0.txt", kernel1, kernel_size * kernel_size * input_channels * channels_conv1);
    read_from_file("weight/weights_NEW_1.txt", kernel2, kernel_size * kernel_size * channels_conv1 * channels_conv2);
    read_from_file("weight/weights_NEW_3.txt", kernel3, kernel_size * kernel_size * channels_conv2 * channels_conv3);
    read_from_file("weight/weights_NEW_4.txt", kernel4, kernel_size * kernel_size * channels_conv3 * channels_conv4);
    read_from_file("weight/weights_NEW_7.txt", weights_fc, 8 * 8 * 64 * output_size);

    // Kết quả sau mỗi lớp Conv2D
    float output_conv1[input_width * input_height  * channels_conv1];
    float output_conv2[input_width  * input_height  * channels_conv2];
    float output_conv2_1[(input_width/2)  * (input_height/2)  * channels_conv3];
    float output_conv3[(input_width/2)  * (input_height/2)  * channels_conv3];
    float output_conv4[(input_width/2) * (input_height/2) * channels_conv4];
    float output_conv4_1[(input_width/4) * (input_height/4) * channels_conv4];

    // Kết quả sau khi chuyển đổi thành dữ liệu cho lớp fully connected
    float fc_output[output_size]; // Đầu ra của lớp Fully Connected
    int correct_predictions = 0;

for(int i = 0; i < 1000; i++){
        char input_filename[50];
        snprintf(input_filename, sizeof(input_filename), "data/image_%d.txt", i); // Đọc từng ảnh từ "data/image_0.txt" đến "data/image_99.txt"
        float input[32 * 32 * 3];
        read_from_file(input_filename, input, 32 * 32 * 3);
        transpose_data(input, input, 32,32,3);
        char label_filename[50];
        snprintf(label_filename, sizeof(label_filename), "data/label_%d.txt", i);
        int true_label = read_label_from_file(label_filename);  // Đọc label từ file
    // Áp dụng các lớp Conv2D với ReLU sau mỗi lớp Conv2D
    conv2d(input, kernel1, bias1, output_conv1, input_width, input_height, input_channels, kernel_size ,kernel_size, channels_conv1, 1, 1 , 1);
    relu(output_conv1, output_conv1, input_width  * input_height  * channels_conv1);
    transpose_data(output_conv1, output_conv1, 32,32,32);
    conv2d(output_conv1, kernel2, bias2, output_conv2, input_width , input_height , channels_conv1, kernel_size, kernel_size, channels_conv2, 1,1,1);
    relu(output_conv2, output_conv2, input_width  * input_height  * channels_conv2);

    // Áp dụng Max Pooling sau mỗi lớp Conv2D
    maxpool(output_conv2, output_conv2_1, input_width, input_height, channels_conv2, 2, 2, 2);
    transpose_data(output_conv2_1, output_conv2_1, 16,16,32);
    conv2d(output_conv2_1, kernel3, bias3, output_conv3, input_width / 2, input_height / 2, channels_conv2, kernel_size, kernel_size, channels_conv3, 1,1,1);
    relu(output_conv3, output_conv3, (input_width/2)  * (input_height/2)  * channels_conv3);
    transpose_data(output_conv3, output_conv3, 16,16,64);
    conv2d(output_conv3, kernel4, bias4, output_conv4, input_width/2 , input_height / 2, channels_conv3, kernel_size, kernel_size, channels_conv4, 1,1,1);
    relu(output_conv4, output_conv4, (input_width / 2) * (input_height / 2) * channels_conv4);

    // Áp dụng Max Pooling sau lớp Conv2D thứ 4
    maxpool(output_conv4, output_conv4_1, input_width / 2, input_height / 2, channels_conv4, 2, 2, 2);

    // Chuyển dữ liệu từ 2D sang 1D để đưa vào lớp Fully Connected
    

    // Áp dụng lớp Fully Connected
    fully_connected(output_conv4_1, weights_fc, fc_output, 8 * 8 * channels_conv4, output_size);
    //relu(fc_output, fc_output, output_size);
    softmax(fc_output, fc_output, output_size);
    int predicted_label = get_max_label(fc_output, output_size);

    // So sánh label dự đoán với label thực tế
    if (predicted_label == true_label) {
        correct_predictions++;  // Tăng số lượng dự đoán đúng
    }
    // printf("Output of the fully connected layer for image %d:\n", i);
    // for (int j = 0; j < output_size; j++) {
    //     printf("%f ", fc_output[j]);
    // }
    //     printf("\n");

    // In kết quả đầu ra của lớp fully connected và in ra label dự đoán
    printf("Image %d - Predicted label: %d, True label: %d\n", i, predicted_label, true_label);
}
    printf("Total correct predictions: %d\n", correct_predictions);
    return 0;
}

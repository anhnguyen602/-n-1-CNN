#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <conv2d.h>
#include <relu.h>
#include <fully_connected.h>

int main() {
    // Các tham số kích thước đầu vào và số lượng lớp Fully Connected
    int input_width = 32, input_height = 32, input_channels = 3;
    int output_size = 10;  // Ví dụ lớp fully connected có 10 đầu ra (phân loại 10 lớp)

    // Các tham số cho kích thước các lớp Conv2D
    int kernel_size = 3;   // Kích thước của kernel (3x3)
    int channels_conv1 = 16, channels_conv2 = 32, channels_conv3 = 64, channels_conv4 = 128;

    // Giả định dữ liệu đầu vào (sử dụng giá trị ngẫu nhiên cho đơn giản)
    float input[32 * 32 * 3] = { /* Dữ liệu đầu vào giả lập */ };

    // Khai báo trọng số và bias cho các lớp Conv2D
    float kernel1[kernel_size * kernel_size * input_channels * channels_conv1];
    float kernel2[kernel_size * kernel_size * channels_conv1 * channels_conv2];
    float kernel3[kernel_size * kernel_size * channels_conv2 * channels_conv3];
    float kernel4[kernel_size * kernel_size * channels_conv3 * channels_conv4];
    float bias1[channels_conv1], bias2[channels_conv2], bias3[channels_conv3], bias4[channels_conv4];

    // Kết quả sau mỗi lớp Conv2D
    float output_conv1[(input_width - kernel_size + 1) * (input_height - kernel_size + 1) * channels_conv1];
    float output_conv2[(input_width - kernel_size * 2 + 2) * (input_height - kernel_size * 2 + 2) * channels_conv2];
    float output_conv3[(input_width - kernel_size * 3 + 3) * (input_height - kernel_size * 3 + 3) * channels_conv3];
    float output_conv4[(input_width - kernel_size * 4 + 4) * (input_height - kernel_size * 4 + 4) * channels_conv4];

    // Kết quả sau khi chuyển đổi thành dữ liệu cho lớp fully connected
    float fc_output[output_size]; // Đầu ra của lớp Fully Connected
    float weights_fc[22 * 22 * 3 * output_size]; // Trọng số lớp fully connected

    // Áp dụng các lớp Conv2D với ReLU sau mỗi lớp Conv2D
    conv2d(input, kernel1, bias1, output_conv1, input_width, input_height, input_channels, kernel_size ,kernel_size, channels_conv1, 1, 1 , 0);
    relu((float*)output_conv1, (float*)output_conv1, (input_width - kernel_size + 1) * (input_height - kernel_size + 1) * channels_conv1);

    conv2d(output_conv1, kernel2, bias2, output_conv2, input_width - kernel_size + 1, input_height - kernel_size + 1, channels_conv1, kernel_size, kernel_size, channels_conv2, 1,1,0);
    relu((float*)output_conv2, (float*)output_conv2, (input_width - kernel_size * 2 + 2) * (input_height - kernel_size * 2 + 2) * channels_conv2);

    conv2d(output_conv2, kernel3, bias3, output_conv3, input_width - kernel_size * 2 + 2, input_height - kernel_size * 2 + 2, channels_conv2, kernel_size, kernel_size, channels_conv3, 1,1,0);
    relu((float*)output_conv3, (float*)output_conv3, (input_width - kernel_size * 3 + 3) * (input_height - kernel_size * 3 + 3) * channels_conv3);

    conv2d(output_conv3, kernel4, bias4, output_conv4, input_width - kernel_size * 3 + 3, input_height - kernel_size * 3 + 3, channels_conv3, kernel_size, kernel_size, channels_conv4, 1,1,0);
    relu((float*)output_conv4, (float*)output_conv4, (input_width - kernel_size * 4 + 4) * (input_height - kernel_size * 4 + 4) * channels_conv4);

    // Áp dụng lớp Fully Connected
    fully_connected(output_conv4, weights_fc, fc_output, 22 * 22 * channels_conv4, output_size);

    // In kết quả đầu ra của lớp fully connected
    printf("Output of the fully connected layer:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", fc_output[i]);
    }
    printf("\n");

    return 0;
}

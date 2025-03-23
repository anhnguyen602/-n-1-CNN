#include <stdio.h>
#include <stdint.h>

void conv2d(
    const int16_t *input,       // Pointer to input data
    const int16_t *kernel,      // Pointer to kernel weights
    const int16_t *bias,        // Pointer to bias (can be NULL)
    int16_t *output,            // Pointer to output data
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
    int padding_width, padding_height;
    if (padding == 0) {
        padding_width = 0;
        padding_height = 0;
    }
    else {
        padding_width = (kernel_width - 1 ) / 2; // same
        padding_height = (kernel_height - 1) / 2; // same
    }
    int output_height = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    int output_width = (input_width - kernel_height + 2 * padding_width) / stride_width + 1;
     for (int oc = 0; oc < output_channels; oc++) {
     // Duyệt qua chiều cao và chiều rộng của đầu ra
     for (int oh = 0; oh < output_height; oh++) {
         for (int ow = 0; ow < output_width; ow++) {
             int32_t value = 0; // Giá trị tích chập cho pixel đầu ra hiện tại
             // Duyệt qua từng kênh đầu vào
             for (int ic = 0; ic < input_channels; ic++) {
                 // Duyệt qua kernel
                 for (int kh = 0; kh < kernel_height; kh++) {
                     for (int kw = 0; kw < kernel_width; kw++) {
                         // Tính toán tọa độ trong đầu vào với padding
                         int ih = oh * stride_height + kh - padding_height;
                         int iw = ow * stride_width + kw - padding_width;

                         // Kiểm tra nếu tọa độ trong phạm vi của input
                         if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                             // Chỉ số trong mảng input
                             int input_idx = (ic * input_height + ih) * input_width + iw;
                             
                             // Chỉ số trong mảng weights
                             // Lưu ý: Chỉ số weight được tính cho mỗi output channel, input channel và kernel
                             int weight_idx = (((oc * input_channels) + ic) * kernel_width + kh) * kernel_height + kw;

                             // Tính tích chập và cộng dồn vào giá trị đầu ra
                             value += input[input_idx] * kernel[weight_idx];
                         }
                     }
                 }
             }
             // Gán giá trị đã tính vào đầu ra
             int output_idx = (oc * output_height + oh) * output_width + ow;
             if (bias != NULL) {
                output[output_idx] = value + bias[oc];
             }
             else {
                output[output_idx] = value;
             }
         }
     }
 }
}

void print_matrix(int16_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // Example 3x3 input with 1 channel
    int16_t input[3 * 3 * 1] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Example 2x2 kernel with 1 channel and 1 output channel
    int16_t kernel[2 * 2 * 1 * 1] = {
        1, 0,
        -1, 1
    };

    // No bias
    int16_t bias[1] = {0};

    // Output array
    int16_t output[2 * 2 * 1];

    // Perform convolution with stride 1, padding 0
    conv2d(input, kernel, bias, output, 3, 3, 1, 2, 2, 1, 1, 1, 0);

    // Print the output
    printf("Output of convolution:\n");
    print_matrix(output, 2, 2); // Output size will be 2x2

    return 0;
}

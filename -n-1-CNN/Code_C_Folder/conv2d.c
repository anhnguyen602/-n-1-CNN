#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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
    int padding_width = padding;
    int padding_height = padding;
    int output_height = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    int output_width = (input_width - kernel_height + 2 * padding_width) / stride_width + 1;
    
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                int32_t value = 0; // Output value for the current pixel
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

// Function to read hex file into an integer array
void read_hex_file(const char *filename, int16_t *data, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File open failed");
        printf("Failed to open file: %s\n", filename);  // Print the filename that failed to open
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        unsigned int value;
        fscanf(file, "%x", &value);
        data[i] = (int16_t)value;  // Store as 16-bit integer
    }
    
    fclose(file);
}

// Function to write the output data to a hex file
void write_output_file(const char *filename, const int16_t *output, int size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("File open failed");
        printf("Failed to open file: %s\n", filename);  // Print the filename that failed to open
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%02X\n", output[i] & 0xFF);  // Write as hex
    }

    fclose(file);
}

int main() {
    // Define the dimensions of the input and kernel
    int input_width = 5, input_height = 5, input_channels = 2;
    int kernel_width = 3, kernel_height = 3, output_channels = 4;
    int stride_width = 1, stride_height = 1, padding = 0;

    // Calculate the size of the input, kernel, and output arrays
    int input_size = input_width * input_height * input_channels;
    int kernel_size = kernel_width * kernel_height * input_channels * output_channels;
    int output_width = (input_width - kernel_width + 2 * padding) / stride_width + 1;
    int output_height = (input_height - kernel_height + 2 * padding) / stride_height + 1;
    int output_size = output_width * output_height * output_channels;

    // Allocate memory for the input, kernel, output, and bias
    int16_t *input = (int16_t *)malloc(input_size * sizeof(int16_t));
    int16_t *kernel = (int16_t *)malloc(kernel_size * sizeof(int16_t));
    int16_t *output = (int16_t *)malloc(output_size * sizeof(int16_t));
    int16_t *bias = (int16_t *)calloc(output_channels, sizeof(int16_t));  // Bias initialized to 0

    // Read the input and kernel data from the hex files
    read_hex_file("C:/Users/Admin/OneDrive - Hanoi University of Science and Technology/Desktop/Do an 1/-n-1-CNN/in-weight-out_golden/IFM.hex", input, input_size);
    read_hex_file("C:/Users/Admin/OneDrive - Hanoi University of Science and Technology/Desktop/Do an 1/-n-1-CNN/in-weight-out_golden/Weight.hex", kernel, kernel_size);

    // Perform the convolution
    conv2d(input, kernel, bias, output, input_width, input_height, input_channels,
           kernel_width, kernel_height, output_channels, stride_width, stride_height, padding);

    // Write the output data to a hex file
    write_output_file("C:/Users/Admin/OneDrive - Hanoi University of Science and Technology/Desktop/Do an 1/-n-1-CNN/in_weight_out_C/OFM.hex", output, output_size);

    // Free allocated memory
    // for (int i = 0; i < kernel_size; i++){
    //     printf("%2X ", kernel[i]);
    // }
    for (int i = 0; i < output_size; i++){
        printf("%2X ", output[i]);
    }
    free(input);
    free(kernel);
    free(output);
    free(bias);
    
    printf("Convolution completed and output written to output.hex\n");

    return 0;
}

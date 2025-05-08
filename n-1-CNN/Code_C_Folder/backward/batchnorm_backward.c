
#include<stdio.h>
#include<stdlib.h>
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

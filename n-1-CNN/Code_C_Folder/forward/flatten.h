
#ifndef FLATTEN_H
#define FLATTEN_H
#include <stdio.h>

#include <stdlib.h>
void flatten_from_hwc_to_whc_flatten(
    float *input, float *output,
    int H, int W, int C
) ;
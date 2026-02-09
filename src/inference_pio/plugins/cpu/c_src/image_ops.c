#include <stdlib.h>
#include <math.h>
#include "../../common/tensor.h"

// Image Resize (Bilinear)
// Input: [C, H, W]
// Output: [C, TargetH, TargetW]
void image_resize_bilinear(float* input, int channels, int h, int w, float* output, int target_h, int target_w) {
    float x_ratio = ((float)(w - 1)) / target_w;
    float y_ratio = ((float)(h - 1)) / target_h;

    #pragma omp parallel for
    for(int c=0; c<channels; c++) {
        for(int i=0; i<target_h; i++) {
            for(int j=0; j<target_w; j++) {
                int x_l = (int)(x_ratio * j);
                int y_l = (int)(y_ratio * i);
                int x_h = (x_l + 1 < w) ? x_l + 1 : x_l;
                int y_h = (y_l + 1 < h) ? y_l + 1 : y_l;

                float x_weight = (x_ratio * j) - x_l;
                float y_weight = (y_ratio * i) - y_l;

                float a = input[(c*h + y_l)*w + x_l];
                float b = input[(c*h + y_l)*w + x_h];
                float cx = input[(c*h + y_h)*w + x_l];
                float d = input[(c*h + y_h)*w + x_h];

                // Bilinear Interpolation
                float val = a * (1 - x_weight) * (1 - y_weight) +
                            b * x_weight * (1 - y_weight) +
                            cx * (1 - x_weight) * y_weight +
                            d * x_weight * y_weight;

                output[(c*target_h + i)*target_w + j] = val;
            }
        }
    }
}

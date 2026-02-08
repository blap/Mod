#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// Basic image resize (Bilinear interpolation)
// Input: [C, H, W] flattened float buffer
// Output: [C, target_H, target_W] flattened float buffer
EXPORT void image_resize_bilinear(float* input, int channels, int h, int w, float* output, int target_h, int target_w) {
    float x_ratio = (float)(w - 1) / target_w;
    float y_ratio = (float)(h - 1) / target_h;

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < target_h; i++) {
            for (int j = 0; j < target_w; j++) {
                int x_l = (int)(x_ratio * j);
                int y_l = (int)(y_ratio * i);
                int x_h = (x_l + 1 < w) ? x_l + 1 : x_l;
                int y_h = (y_l + 1 < h) ? y_l + 1 : y_l;

                float x_weight = (x_ratio * j) - x_l;
                float y_weight = (y_ratio * i) - y_l;

                float a = input[c * h * w + y_l * w + x_l];
                float b = input[c * h * w + y_l * w + x_h];
                float k = input[c * h * w + y_h * w + x_l];
                float d = input[c * h * w + y_h * w + x_h];

                float pixel = a * (1 - x_weight) * (1 - y_weight) +
                              b * x_weight * (1 - y_weight) +
                              k * y_weight * (1 - x_weight) +
                              d * x_weight * y_weight;

                output[c * target_h * target_w + i * target_w + j] = pixel;
            }
        }
    }
}

// Image Normalize
// Input: [C, H, W]
// Output: (input - mean) / std per channel
EXPORT void image_normalize(float* image, int channels, int h, int w, float* mean, float* std) {
    int pixels = h * w;
    for (int c = 0; c < channels; c++) {
        float m = mean[c];
        float s = std[c];
        for (int i = 0; i < pixels; i++) {
            int idx = c * pixels + i;
            image[idx] = (image[idx] - m) / s;
        }
    }
}

// Image Rescale
EXPORT void image_rescale(float* image, int size, float scale) {
    for (int i = 0; i < size; i++) {
        image[i] *= scale;
    }
}

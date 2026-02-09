#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../common/tensor.h"

// Minimal SafeTensors Loader (Dependency-Free)
// Supports reading tensors from a .safetensors file by name.

FILE* g_current_file = NULL;
char* g_header_json = NULL;
uint64_t g_header_size = 0;

int open_safetensors(const char* filepath) {
    if (g_current_file) fclose(g_current_file);
    g_current_file = fopen(filepath, "rb");
    if (!g_current_file) return 0;

    // Read header size (8 bytes, little endian uint64)
    if (fread(&g_header_size, 8, 1, g_current_file) != 1) {
        fclose(g_current_file); g_current_file = NULL; return 0;
    }

    // Read header JSON
    if (g_header_json) free(g_header_json);
    g_header_json = (char*)malloc(g_header_size + 1);
    if (fread(g_header_json, 1, g_header_size, g_current_file) != g_header_size) {
        free(g_header_json); g_header_json = NULL;
        fclose(g_current_file); g_current_file = NULL; return 0;
    }
    g_header_json[g_header_size] = '\0';
    return 1;
}

// Simple string search for "tensor_name": { ... "data_offsets": [start, end] ... }
// This is fragile but works for standard safetensors if keys are quoted.
int load_tensor_data(const char* name, float* buffer, int size) {
    if (!g_current_file || !g_header_json) return 0;

    char search_key[256];
    snprintf(search_key, 256, "\"%s\"", name);

    char* pos = strstr(g_header_json, search_key);
    if (!pos) return 0;

    // Found name. Find "data_offsets" after it.
    char* offsets_key = "\"data_offsets\"";
    char* offsets_pos = strstr(pos, offsets_key);
    if (!offsets_pos) return 0;

    // Parse [start, end]
    char* bracket_start = strchr(offsets_pos, '[');
    if (!bracket_start) return 0;

    long start, end;
    if (sscanf(bracket_start + 1, "%ld, %ld", &start, &end) != 2) return 0;

    // Seek and read
    long data_start = 8 + g_header_size + start;
    fseek(g_current_file, data_start, SEEK_SET);

    // Check size (bytes vs float elements)
    long bytes = end - start;
    if (bytes == size * sizeof(float)) {
        if (fread(buffer, sizeof(float), size, g_current_file) != (size_t)size) return 0;
    } else if (bytes == size * sizeof(uint16_t)) {
        // Handle fp16 -> fp32 conversion
        uint16_t* temp = (uint16_t*)malloc(bytes);
        if (fread(temp, 1, bytes, g_current_file) != (size_t)bytes) { free(temp); return 0; }

        // Simple fp16 to fp32 (ignoring denormals/inf/nan handling for brevity or using library)
        // Since no library, we use a simple conversion or just cast if compiler supports _Float16
        // or shift bits.
        // Fast approx: extract exponent and mantissa.
        for(int i=0; i<size; i++) {
            uint16_t h = temp[i];
            uint32_t s = (h >> 15) & 0x1;
            uint32_t e = (h >> 10) & 0x1F;
            uint32_t m = h & 0x3FF;

            uint32_t f;
            if (e == 0) {
                if (m == 0) f = s << 31;
                else {
                    // Denormal
                    while (!(m & 0x400)) {
                        m <<= 1;
                        e--;
                    }
                    e++;
                    m &= ~0x400;
                    f = (s << 31) | ((e + 112) << 23) | (m << 13);
                }
            } else if (e == 31) {
                if (m == 0) f = (s << 31) | 0x7F800000; // Inf
                else f = (s << 31) | 0x7F800000 | (m << 13); // NaN
            } else {
                f = (s << 31) | ((e + 112) << 23) | (m << 13);
            }

            memcpy(&buffer[i], &f, sizeof(float));
        }
        free(temp);
    } else {
        return 0; // Size mismatch / unknown type
    }

    return 1;
}

void close_safetensors() {
    if (g_current_file) { fclose(g_current_file); g_current_file = NULL; }
    if (g_header_json) { free(g_header_json); g_header_json = NULL; }
}

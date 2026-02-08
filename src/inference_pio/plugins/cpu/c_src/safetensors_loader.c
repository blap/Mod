#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// Minimal JSON parser helper (very basic, assumes standard SafeTensors header format)
// Returns the offset to the start of binary data and fills tensor info map (simplified)

typedef struct {
    char name[256];
    size_t shape[8];
    int ndim;
    size_t data_offset[2]; // start, end
    char dtype[16];
} TensorInfo;

// Simplified loader that reads the header size and finds tensor offsets
// In a real implementation, we would use a JSON library like cJSON or JSMN.
// Here we implement a naive parser sufficient for the task without external deps.

uint64_t read_u64(FILE* f) {
    uint64_t val;
    if (fread(&val, sizeof(uint64_t), 1, f) != 1) return 0;
    return val; // SafeTensors is little-endian
}

// Global buffer for loaded tensors metadata (simplified storage)
#define MAX_TENSORS 1024
TensorInfo g_tensor_registry[MAX_TENSORS];
int g_tensor_count = 0;
FILE* g_current_file = NULL;
uint64_t g_header_size = 0;

EXPORT void close_safetensors() {
    if (g_current_file) {
        fclose(g_current_file);
        g_current_file = NULL;
    }
    g_tensor_count = 0;
}

EXPORT int open_safetensors(const char* filepath) {
    close_safetensors();
    g_current_file = fopen(filepath, "rb");
    if (!g_current_file) return -1;

    // Read header size (8 bytes, little endian)
    g_header_size = read_u64(g_current_file);
    if (g_header_size == 0) return -2;

    // Read header JSON string
    char* header_json = (char*)malloc(g_header_size + 1);
    if (fread(header_json, 1, g_header_size, g_current_file) != g_header_size) {
        free(header_json);
        return -3;
    }
    header_json[g_header_size] = '\0';

    // Parse JSON (Naive regex-like scan)

    char* cursor = header_json;
    while (*cursor && g_tensor_count < MAX_TENSORS) {
        // Find next key (tensor name)
        char* key_start = strchr(cursor, '\"');
        if (!key_start) break;
        key_start++;
        char* key_end = strchr(key_start, '\"');
        if (!key_end) break;

        // Check if this is metadata key "__metadata__"
        if (strncmp(key_start, "__metadata__", 12) == 0) {
            cursor = strchr(key_end + 1, '}'); // Skip object
            if (!cursor) break;
            cursor++;
            continue;
        }

        // Store Name
        int name_len = key_end - key_start;
        if (name_len > 255) name_len = 255;
        strncpy(g_tensor_registry[g_tensor_count].name, key_start, name_len);
        g_tensor_registry[g_tensor_count].name[name_len] = '\0';

        // Find data_offsets
        char* offsets_key = strstr(key_end, "\"data_offsets\"");
        if (!offsets_key) break;
        char* bracket_start = strchr(offsets_key, '[');
        if (!bracket_start) break;

        long start = strtol(bracket_start + 1, &cursor, 10);
        while (*cursor == ' ' || *cursor == ',') cursor++;
        long end = strtol(cursor, &cursor, 10);

        g_tensor_registry[g_tensor_count].data_offset[0] = start;
        g_tensor_registry[g_tensor_count].data_offset[1] = end;

        g_tensor_count++;

        // Find end of this object
        char* obj_end = strchr(key_end, '}');
        if (!obj_end) break;
        cursor = obj_end + 1;
    }

    free(header_json);
    return g_tensor_count;
}

// Load tensor data by name into a pre-allocated float buffer
// Returns 0 on success, -1 on failure/not found
EXPORT int load_tensor_data(const char* name, float* buffer, int size) {
    if (!g_current_file) return -1;

    for (int i = 0; i < g_tensor_count; i++) {
        if (strcmp(g_tensor_registry[i].name, name) == 0) {
            long start = g_tensor_registry[i].data_offset[0];
            long end = g_tensor_registry[i].data_offset[1];
            long bytes = end - start;

            // Seek to data (Header Size + 8 bytes length + offset)
            long file_offset = 8 + g_header_size + start;
            fseek(g_current_file, file_offset, SEEK_SET);

            if (bytes == size * sizeof(float)) {
                // Direct read (FP32)
                fread(buffer, sizeof(float), size, g_current_file);
            } else if (bytes == size * sizeof(uint16_t)) {
                // Convert FP16 to FP32 on the fly
                uint16_t* temp = (uint16_t*)malloc(bytes);
                fread(temp, 1, bytes, g_current_file);

                #pragma omp parallel for
                for (int j = 0; j < size; j++) {
                    uint16_t h = temp[j];
                    uint16_t h_exp = (h >> 10) & 0x1f;
                    uint16_t h_sig = h & 0x3ff;

                    uint32_t f_sign = (h >> 15) << 31;
                    uint32_t f_exp = (h_exp + 112) << 23;
                    uint32_t f_sig = h_sig << 13;

                    if (h_exp == 0) {
                        if (h_sig == 0) f_exp = 0;
                        else f_exp = 0;
                    } else if (h_exp == 0x1f) {
                        f_exp = 0xff << 23;
                    }

                    uint32_t f_bits = f_sign | f_exp | f_sig;
                    memcpy(&buffer[j], &f_bits, sizeof(float));
                }
                free(temp);
            }

            return 0;
        }
    }
    return -1; // Not found
}

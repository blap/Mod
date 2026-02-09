#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../common/tensor.h"

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define USE_MMAP
#elif defined(_WIN32)
#include <windows.h>
#define USE_MMAP_WIN
#endif

// Minimal SafeTensors Loader (Dependency-Free)
// Supports reading tensors from a .safetensors file by name.

FILE* g_current_file = NULL;
char* g_header_json = NULL;
uint64_t g_header_size = 0;

#ifdef USE_MMAP
int g_fd = -1;
void* g_mmap_addr = NULL;
size_t g_mmap_size = 0;
#endif

#ifdef USE_MMAP_WIN
HANDLE g_hFile = INVALID_HANDLE_VALUE;
HANDLE g_hMap = NULL;
void* g_mmap_addr_win = NULL;
size_t g_mmap_size_win = 0;
#endif

int open_safetensors(const char* filepath) {
    if (g_current_file) fclose(g_current_file);
    #ifdef USE_MMAP
    if (g_mmap_addr) { munmap(g_mmap_addr, g_mmap_size); g_mmap_addr = NULL; }
    if (g_fd != -1) { close(g_fd); g_fd = -1; }
    #endif
    #ifdef USE_MMAP_WIN
    if (g_mmap_addr_win) { UnmapViewOfFile(g_mmap_addr_win); g_mmap_addr_win = NULL; }
    if (g_hMap) { CloseHandle(g_hMap); g_hMap = NULL; }
    if (g_hFile != INVALID_HANDLE_VALUE) { CloseHandle(g_hFile); g_hFile = INVALID_HANDLE_VALUE; }
    #endif

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

    #ifdef USE_MMAP
    // Linux Mmap
    g_fd = open(filepath, O_RDONLY);
    if (g_fd != -1) {
        struct stat sb;
        if (fstat(g_fd, &sb) != -1) {
            g_mmap_size = sb.st_size;
            g_mmap_addr = mmap(NULL, g_mmap_size, PROT_READ, MAP_PRIVATE, g_fd, 0);
            if (g_mmap_addr == MAP_FAILED) {
                g_mmap_addr = NULL;
                close(g_fd); g_fd = -1;
            }
        } else {
            close(g_fd); g_fd = -1;
        }
    }
    #endif

    #ifdef USE_MMAP_WIN
    // Windows Map
    g_hFile = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_hFile != INVALID_HANDLE_VALUE) {
        DWORD high = 0;
        DWORD low = GetFileSize(g_hFile, &high);
        g_mmap_size_win = ((size_t)high << 32) | low;

        g_hMap = CreateFileMappingA(g_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (g_hMap) {
            g_mmap_addr_win = MapViewOfFile(g_hMap, FILE_MAP_READ, 0, 0, 0);
            if (!g_mmap_addr_win) {
                CloseHandle(g_hMap); g_hMap = NULL;
                CloseHandle(g_hFile); g_hFile = INVALID_HANDLE_VALUE;
            }
        } else {
            CloseHandle(g_hFile); g_hFile = INVALID_HANDLE_VALUE;
        }
    }
    #endif

    return 1;
}

// Helper for F16->F32
static inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 0x1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else {
            while (!(m & 0x400)) { m <<= 1; e--; }
            e++;
            m &= ~0x400;
            f = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        if (m == 0) f = (s << 31) | 0x7F800000;
        else f = (s << 31) | 0x7F800000 | (m << 13);
    } else {
        f = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Simple string search for "tensor_name": { ... "data_offsets": [start, end] ... }
// This is fragile but works for standard safetensors if keys are quoted.
int load_tensor_data(const char* name, float* buffer, int size) {
    int mapped = 0;
    #ifdef USE_MMAP
    if (g_mmap_addr) mapped = 1;
    #endif
    #ifdef USE_MMAP_WIN
    if (g_mmap_addr_win) mapped = 1;
    #endif

    if ((!g_current_file && !mapped) || !g_header_json) return 0;

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

    int64_t start, end;
    // Use %ld for Linux 64-bit long, but safest is %lld and casting or int64 specific macros.
    // Assuming standard C library support for %lld for int64_t (C99).
    if (sscanf(bracket_start + 1, "%lld, %lld", (long long*)&start, (long long*)&end) != 2) return 0;

    int64_t data_start = 8 + g_header_size + start;
    int64_t bytes = end - start;

    #ifdef USE_MMAP
    if (g_mmap_addr) {
        if (data_start + bytes > (int64_t)g_mmap_size) return 0;
        uint8_t* ptr = (uint8_t*)g_mmap_addr + data_start;

        if (bytes == size * sizeof(float)) {
            memcpy(buffer, ptr, bytes);
        } else if (bytes == size * sizeof(uint16_t)) {
            uint16_t* temp = (uint16_t*)ptr;
            #pragma omp parallel for
            for(int i=0; i<size; i++) {
                buffer[i] = half_to_float(temp[i]);
            }
        } else {
            return 0;
        }
        return 1;
    }
    #endif

    #ifdef USE_MMAP_WIN
    if (g_mmap_addr_win) {
        if (data_start + bytes > (int64_t)g_mmap_size_win) return 0;
        uint8_t* ptr = (uint8_t*)g_mmap_addr_win + data_start;

        if (bytes == size * sizeof(float)) {
            memcpy(buffer, ptr, bytes);
        } else if (bytes == size * sizeof(uint16_t)) {
            uint16_t* temp = (uint16_t*)ptr;
            #pragma omp parallel for
            for(int i=0; i<size; i++) {
                buffer[i] = half_to_float(temp[i]);
            }
        } else {
            return 0;
        }
        return 1;
    }
    #endif

    // Fallback to fread
    #ifdef _WIN32
    _fseeki64(g_current_file, data_start, SEEK_SET);
    #else
    fseeko(g_current_file, data_start, SEEK_SET);
    #endif

    if (bytes == size * sizeof(float)) {
        if (fread(buffer, sizeof(float), size, g_current_file) != (size_t)size) return 0;
    } else if (bytes == size * sizeof(uint16_t)) {
        uint16_t* temp = (uint16_t*)malloc(bytes);
        if (fread(temp, 1, bytes, g_current_file) != (size_t)bytes) { free(temp); return 0; }

        #pragma omp parallel for
        for(int i=0; i<size; i++) {
            buffer[i] = half_to_float(temp[i]);
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
    #ifdef USE_MMAP
    if (g_mmap_addr) { munmap(g_mmap_addr, g_mmap_size); g_mmap_addr = NULL; }
    if (g_fd != -1) { close(g_fd); g_fd = -1; }
    #endif
    #ifdef USE_MMAP_WIN
    if (g_mmap_addr_win) { UnmapViewOfFile(g_mmap_addr_win); g_mmap_addr_win = NULL; }
    if (g_hMap) { CloseHandle(g_hMap); g_hMap = NULL; }
    if (g_hFile != INVALID_HANDLE_VALUE) { CloseHandle(g_hFile); g_hFile = INVALID_HANDLE_VALUE; }
    #endif
}

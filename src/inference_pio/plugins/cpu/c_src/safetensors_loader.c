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

// SafeTensors Loader - Context Based (Thread Safe / Re-entrant)

struct SafetensorsContext {
    FILE* file;
    char* header_json;
    uint64_t header_size;

    #ifdef USE_MMAP
    int fd;
    void* mmap_addr;
    size_t mmap_size;
    #endif

    #ifdef USE_MMAP_WIN
    HANDLE hFile;
    HANDLE hMap;
    void* mmap_addr_win;
    size_t mmap_size_win;
    #endif
};

EXPORT SafetensorsContext* open_safetensors(const char* filepath) {
    SafetensorsContext* ctx = (SafetensorsContext*)calloc(1, sizeof(SafetensorsContext));
    if (!ctx) return NULL;

    #ifdef USE_MMAP
    ctx->fd = -1;
    #endif
    #ifdef USE_MMAP_WIN
    ctx->hFile = INVALID_HANDLE_VALUE;
    #endif

    ctx->file = fopen(filepath, "rb");
    if (!ctx->file) {
        free(ctx);
        return NULL;
    }

    // Read header size
    if (fread(&ctx->header_size, 8, 1, ctx->file) != 1) {
        fclose(ctx->file);
        free(ctx);
        return NULL;
    }

    // Read header JSON
    ctx->header_json = (char*)malloc(ctx->header_size + 1);
    if (!ctx->header_json) {
        fclose(ctx->file);
        free(ctx);
        return NULL;
    }
    if (fread(ctx->header_json, 1, ctx->header_size, ctx->file) != ctx->header_size) {
        free(ctx->header_json);
        fclose(ctx->file);
        free(ctx);
        return NULL;
    }
    ctx->header_json[ctx->header_size] = '\0';

    #ifdef USE_MMAP
    // Linux Mmap
    ctx->fd = open(filepath, O_RDONLY);
    if (ctx->fd != -1) {
        struct stat sb;
        if (fstat(ctx->fd, &sb) != -1) {
            ctx->mmap_size = sb.st_size;
            ctx->mmap_addr = mmap(NULL, ctx->mmap_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
            if (ctx->mmap_addr == MAP_FAILED) {
                ctx->mmap_addr = NULL;
                close(ctx->fd); ctx->fd = -1;
            }
        } else {
            close(ctx->fd); ctx->fd = -1;
        }
    }
    #endif

    #ifdef USE_MMAP_WIN
    // Windows Map
    ctx->hFile = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (ctx->hFile != INVALID_HANDLE_VALUE) {
        DWORD high = 0;
        DWORD low = GetFileSize(ctx->hFile, &high);
        ctx->mmap_size_win = ((size_t)high << 32) | low;

        ctx->hMap = CreateFileMappingA(ctx->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (ctx->hMap) {
            ctx->mmap_addr_win = MapViewOfFile(ctx->hMap, FILE_MAP_READ, 0, 0, 0);
            if (!ctx->mmap_addr_win) {
                CloseHandle(ctx->hMap); ctx->hMap = NULL;
                CloseHandle(ctx->hFile); ctx->hFile = INVALID_HANDLE_VALUE;
            }
        } else {
            CloseHandle(ctx->hFile); ctx->hFile = INVALID_HANDLE_VALUE;
        }
    }
    #endif

    return ctx;
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

EXPORT int load_tensor_data(SafetensorsContext* ctx, const char* name, float* buffer, int size) {
    if (!ctx || !ctx->header_json) return 0;

    int mapped = 0;
    #ifdef USE_MMAP
    if (ctx->mmap_addr) mapped = 1;
    #endif
    #ifdef USE_MMAP_WIN
    if (ctx->mmap_addr_win) mapped = 1;
    #endif

    char search_key[256];
    snprintf(search_key, 256, "\"%s\"", name);

    char* pos = strstr(ctx->header_json, search_key);
    if (!pos) return 0;

    // Found name. Find "data_offsets" after it.
    char* offsets_key = "\"data_offsets\"";
    char* offsets_pos = strstr(pos, offsets_key);
    if (!offsets_pos) return 0;

    // Parse [start, end]
    char* bracket_start = strchr(offsets_pos, '[');
    if (!bracket_start) return 0;

    int64_t start, end;
    if (sscanf(bracket_start + 1, "%lld, %lld", (long long*)&start, (long long*)&end) != 2) return 0;

    int64_t data_start = 8 + ctx->header_size + start;
    int64_t bytes = end - start;

    #ifdef USE_MMAP
    if (ctx->mmap_addr) {
        if (data_start + bytes > (int64_t)ctx->mmap_size) return 0;
        uint8_t* ptr = (uint8_t*)ctx->mmap_addr + data_start;

        if (bytes == size * (int64_t)sizeof(float)) {
            memcpy(buffer, ptr, bytes);
        } else if (bytes == size * (int64_t)sizeof(uint16_t)) {
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
    if (ctx->mmap_addr_win) {
        if (data_start + bytes > (int64_t)ctx->mmap_size_win) return 0;
        uint8_t* ptr = (uint8_t*)ctx->mmap_addr_win + data_start;

        if (bytes == size * (int64_t)sizeof(float)) {
            memcpy(buffer, ptr, bytes);
        } else if (bytes == size * (int64_t)sizeof(uint16_t)) {
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
    if (!ctx->file) return 0;

    #ifdef _WIN32
    _fseeki64(ctx->file, data_start, SEEK_SET);
    #else
    fseeko(ctx->file, data_start, SEEK_SET);
    #endif

    if (bytes == size * (int64_t)sizeof(float)) {
        if (fread(buffer, sizeof(float), size, ctx->file) != (size_t)size) return 0;
    } else if (bytes == size * (int64_t)sizeof(uint16_t)) {
        uint16_t* temp = (uint16_t*)malloc(bytes);
        if (fread(temp, 1, bytes, ctx->file) != (size_t)bytes) { free(temp); return 0; }

        #pragma omp parallel for
        for(int i=0; i<size; i++) {
            buffer[i] = half_to_float(temp[i]);
        }
        free(temp);
    } else {
        return 0;
    }

    return 1;
}

EXPORT void close_safetensors(SafetensorsContext* ctx) {
    if (!ctx) return;

    if (ctx->file) { fclose(ctx->file); ctx->file = NULL; }
    if (ctx->header_json) { free(ctx->header_json); ctx->header_json = NULL; }

    #ifdef USE_MMAP
    if (ctx->mmap_addr) { munmap(ctx->mmap_addr, ctx->mmap_size); ctx->mmap_addr = NULL; }
    if (ctx->fd != -1) { close(ctx->fd); ctx->fd = -1; }
    #endif

    #ifdef USE_MMAP_WIN
    if (ctx->mmap_addr_win) { UnmapViewOfFile(ctx->mmap_addr_win); ctx->mmap_addr_win = NULL; }
    if (ctx->hMap) { CloseHandle(ctx->hMap); ctx->hMap = NULL; }
    if (ctx->hFile != INVALID_HANDLE_VALUE) { CloseHandle(ctx->hFile); ctx->hFile = INVALID_HANDLE_VALUE; }
    #endif

    free(ctx);
}

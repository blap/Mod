@echo off
REM Windows Build Script for Native CPU Plugin using MinGW or MSVC
REM Currently set up for GCC (MinGW)

echo Building CPU Plugin (libtensor_ops.dll)...
gcc -shared -O3 -Wall -Wextra -fopenmp -D_WIN32 -o libtensor_ops.dll tensor_ops.c safetensors_loader.c image_ops.c -lm -I../../common

IF %ERRORLEVEL% EQU 0 (
    echo Build Successful: libtensor_ops.dll
) ELSE (
    echo Build Failed
)

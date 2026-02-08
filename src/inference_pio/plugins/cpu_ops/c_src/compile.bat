@echo off
REM Windows Build Script for CPU Ops Plugin using MinGW or MSVC
REM Currently set up for GCC (MinGW)

gcc -shared -O3 -Wall -Wextra -fopenmp -D_WIN32 -o libtensor_ops.dll tensor_ops.c safetensors_loader.c -lm

IF %ERRORLEVEL% EQU 0 (
    echo Build Successful: libtensor_ops.dll
) ELSE (
    echo Build Failed
)

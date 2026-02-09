@echo off
REM Windows Build Script for CUDA Plugin using MSVC/NVCC
REM Requires CUDA Toolkit and VS Build Tools

echo Building CUDA Plugin (libtensor_ops_cuda.dll)...
nvcc -shared -O3 -o libtensor_ops_cuda.dll tensor_ops_cuda.cu -I../../common

IF %ERRORLEVEL% EQU 0 (
    echo Build Successful: libtensor_ops_cuda.dll
) ELSE (
    echo Build Failed
)

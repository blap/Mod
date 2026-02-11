# Windows Compile Script for CUDA
nvcc -shared -o libtensor_ops_cuda.dll tensor_ops_cuda.cu -Xcompiler -fPIC -O3

# SM61 Specific
nvcc -shared -o libtensor_ops_cuda_sm61.dll tensor_ops_cuda.cu -arch=sm_61 -Xcompiler -fPIC -O3


import ctypes
import os
import logging
from ..base.gpu_interface import GPUHardwareInterface

logger = logging.getLogger(__name__)

# OpenCL Constants
CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 4
CL_MEM_READ_WRITE = 1
CL_MEM_COPY_HOST_PTR = 32

class OpenCLBackend(GPUHardwareInterface):
    """
    Shared OpenCL Backend for AMD and Intel GPUs.
    Uses ctypes to load libOpenCL.so and JIT compile kernels at runtime.
    """
    def __init__(self, platform_vendor_filter=None):
        self.lib = None
        self.ctx = None
        self.queue = None
        self.program = None
        self.kernels = {}
        self.use_cpu_fallback = False
        self.platform_vendor_filter = platform_vendor_filter

        self._load_library()

    def _load_library(self):
        try:
            lib_names = ["libOpenCL.so.1", "libOpenCL.so", "OpenCL.dll"]
            for name in lib_names:
                try:
                    self.lib = ctypes.CDLL(name)
                    break
                except OSError: continue

            if not self.lib:
                raise OSError("OpenCL library not found")

            # Setup ctypes signatures
            self.lib.clGetPlatformIDs.restype = ctypes.c_int
            self.lib.clGetPlatformInfo.restype = ctypes.c_int
            self.lib.clGetDeviceIDs.restype = ctypes.c_int
            self.lib.clCreateContext.restype = ctypes.c_void_p
            self.lib.clCreateCommandQueue.restype = ctypes.c_void_p
            self.lib.clCreateBuffer.restype = ctypes.c_void_p
            self.lib.clEnqueueWriteBuffer.restype = ctypes.c_int
            self.lib.clEnqueueReadBuffer.restype = ctypes.c_int
            self.lib.clCreateProgramWithSource.restype = ctypes.c_void_p
            self.lib.clBuildProgram.restype = ctypes.c_int
            self.lib.clCreateKernel.restype = ctypes.c_void_p
            self.lib.clSetKernelArg.restype = ctypes.c_int
            self.lib.clEnqueueNDRangeKernel.restype = ctypes.c_int
            self.lib.clReleaseMemObject.restype = ctypes.c_int
            self.lib.clFinish.restype = ctypes.c_int
            self.lib.clReleaseContext.restype = ctypes.c_int

        except Exception as e:
            logger.warning(f"OpenCL Init Failed: {e}. Using CPU Fallback.")
            self.use_cpu_fallback = True

    def initialize(self, **kwargs) -> bool:
        if self.use_cpu_fallback: return True

        try:
            # 1. Platform Selection
            num_platforms = ctypes.c_uint()
            self.lib.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
            if num_platforms.value == 0: raise RuntimeError("No OpenCL platforms found")

            platforms = (ctypes.c_void_p * num_platforms.value)()
            self.lib.clGetPlatformIDs(num_platforms.value, platforms, None)

            selected_platform = platforms[0]

            # Filter by vendor if requested (e.g. "AMD", "Intel")
            if self.platform_vendor_filter:
                found = False
                buf = ctypes.create_string_buffer(128)
                for p in platforms:
                    # CL_PLATFORM_VENDOR = 0x0903
                    self.lib.clGetPlatformInfo(p, 0x0903, 128, buf, None)
                    vendor = buf.value.decode('utf-8', 'ignore')
                    if self.platform_vendor_filter.lower() in vendor.lower():
                        selected_platform = p
                        found = True
                        break
                if not found:
                    logger.warning(f"OpenCL platform matching '{self.platform_vendor_filter}' not found. Using default.")

            # 2. Device
            device_id = ctypes.c_void_p()
            num_devices = ctypes.c_uint()
            ret = self.lib.clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, ctypes.byref(device_id), ctypes.byref(num_devices))
            if ret != CL_SUCCESS:
                logger.warning("No OpenCL GPU found on platform. Switching to CPU Fallback.")
                self.use_cpu_fallback = True
                return True

            # 3. Context & Queue
            self.ctx = self.lib.clCreateContext(None, 1, ctypes.byref(device_id), None, None, ctypes.byref(ctypes.c_int()))
            self.queue = self.lib.clCreateCommandQueue(self.ctx, device_id, 0, ctypes.byref(ctypes.c_int()))

            # 4. Compile Kernels
            self._compile_kernels()

            return True
        except Exception as e:
            logger.error(f"OpenCL Initialization Error: {e}")
            self.use_cpu_fallback = True
            return True

    def _compile_kernels(self):
        source = """
        __kernel void matmul(const int M, const int N, const int K,
                             __global const float* A,
                             __global const float* B,
                             __global float* C) {
            int row = get_global_id(1);
            int col = get_global_id(0);

            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k=0; k<K; k++) {
                    sum += A[row*K + k] * B[k*N + col];
                }
                C[row*N + col] = sum;
            }
        }
        """
        c_src = ctypes.create_string_buffer(source.encode('utf-8'))
        src_ptr = ctypes.cast(ctypes.pointer(c_src), ctypes.POINTER(ctypes.c_char_p))

        self.program = self.lib.clCreateProgramWithSource(self.ctx, 1, src_ptr, None, ctypes.byref(ctypes.c_int()))
        ret = self.lib.clBuildProgram(self.program, 0, None, None, None, None)
        if ret != CL_SUCCESS:
            logger.error(f"OpenCL Build Failed: {ret}")
            return

        self.kernels['matmul'] = self.lib.clCreateKernel(self.program, b"matmul", ctypes.byref(ctypes.c_int()))

    def get_device_info(self) -> dict:
        mode = "OpenCL GPU" if not self.use_cpu_fallback else "CPU Fallback"
        return {"backend": mode, "status": "Ready", "vendor_filter": self.platform_vendor_filter}

    def allocate(self, size_bytes: int):
        if self.use_cpu_fallback:
            return ctypes.cast(ctypes.create_string_buffer(size_bytes), ctypes.c_void_p)
        err = ctypes.c_int()
        mem = self.lib.clCreateBuffer(self.ctx, CL_MEM_READ_WRITE, size_bytes, None, ctypes.byref(err))
        return mem

    def free(self, ptr):
        if self.use_cpu_fallback: return
        if ptr: self.lib.clReleaseMemObject(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        if self.use_cpu_fallback:
            ctypes.memmove(dst_ptr, (ctypes.c_float * len(src_data))(*src_data), size_bytes)
            return
        FloatArray = ctypes.c_float * (size_bytes // 4)
        c_data = FloatArray(*src_data)
        self.lib.clEnqueueWriteBuffer(self.queue, dst_ptr, 1, 0, size_bytes, c_data, 0, None, None)

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        if self.use_cpu_fallback:
            c_float_ptr = ctypes.cast(src_ptr, ctypes.POINTER(ctypes.c_float))
            dst_data[:] = [c_float_ptr[i] for i in range(size_bytes//4)]
            return
        FloatArray = ctypes.c_float * (size_bytes // 4)
        c_data = FloatArray()
        self.lib.clEnqueueReadBuffer(self.queue, src_ptr, 1, 0, size_bytes, c_data, 0, None, None)
        dst_data[:] = list(c_data)

    def matmul(self, a_ptr, b_ptr, c_ptr, M, N, K):
        if self.use_cpu_fallback: pass
        else:
            kernel = self.kernels.get('matmul')
            if kernel:
                self.lib.clSetKernelArg(kernel, 0, 4, ctypes.byref(ctypes.c_int(M)))
                self.lib.clSetKernelArg(kernel, 1, 4, ctypes.byref(ctypes.c_int(N)))
                self.lib.clSetKernelArg(kernel, 2, 4, ctypes.byref(ctypes.c_int(K)))
                self.lib.clSetKernelArg(kernel, 3, 8, ctypes.byref(a_ptr))
                self.lib.clSetKernelArg(kernel, 4, 8, ctypes.byref(b_ptr))
                self.lib.clSetKernelArg(kernel, 5, 8, ctypes.byref(c_ptr))
                global_work_size = (ctypes.c_size_t * 2)(N, M)
                self.lib.clEnqueueNDRangeKernel(self.queue, kernel, 2, None, global_work_size, None, 0, None, None)
                self.lib.clFinish(self.queue)

    def synchronize(self):
        if self.queue: self.lib.clFinish(self.queue)

    def cleanup(self):
        if self.ctx: self.lib.clReleaseContext(self.ctx)

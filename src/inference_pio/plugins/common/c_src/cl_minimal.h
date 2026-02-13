#ifndef CL_MINIMAL_H
#define CL_MINIMAL_H

#include <stdint.h>

/* Minimal OpenCL Definitions to avoid system headers dependency */

typedef int cl_int;
typedef unsigned int cl_uint;
typedef uint64_t cl_ulong;
typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;

#define CL_SUCCESS                                  0
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_PLATFORM_VENDOR                          0x0903

#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_USE_HOST_PTR                         (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)

#define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)
#define CL_PROFILING_COMMAND_QUEUED                 0x1280
#define CL_PROFILING_COMMAND_SUBMIT                 0x1281
#define CL_PROFILING_COMMAND_START                  0x1282
#define CL_PROFILING_COMMAND_END                    0x1283

#define CL_TRUE                                     1
#define CL_FALSE                                    0

// Function Pointer Types for Dynamic Loading
typedef cl_int (*PTR_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*PTR_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*PTR_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*PTR_clCreateContext)(const void*, cl_uint, const cl_device_id*, void(*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (*PTR_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (*PTR_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_int (*PTR_clReleaseMemObject)(cl_mem);
typedef cl_int (*PTR_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_int, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PTR_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_int, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_program (*PTR_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*PTR_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void(*)(cl_program, void*), void*);
typedef cl_kernel (*PTR_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*PTR_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*PTR_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PTR_clFinish)(cl_command_queue);
typedef cl_int (*PTR_clReleaseKernel)(cl_kernel);
typedef cl_int (*PTR_clReleaseProgram)(cl_program);
typedef cl_int (*PTR_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*PTR_clReleaseContext)(cl_context);
typedef cl_int (*PTR_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*PTR_clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PTR_clGetProgramInfo)(cl_program, cl_uint, size_t, void*, size_t*);
typedef cl_program (*PTR_clCreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id*, const size_t*, const unsigned char**, cl_int*, cl_int*);
typedef cl_int (*PTR_clWaitForEvents)(cl_uint, const cl_event*);
typedef cl_int (*PTR_clReleaseEvent)(cl_event);
typedef cl_int (*PTR_clGetEventProfilingInfo)(cl_event, cl_uint, size_t, void*, size_t*);
typedef cl_int (*PTR_clEnqueueFillBuffer)(cl_command_queue, cl_mem, const void*, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);

#endif // CL_MINIMAL_H

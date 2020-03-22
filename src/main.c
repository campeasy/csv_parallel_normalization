/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    main.c
    CSV file parallel normalization - program entry point
*/

#include "libs/ocl_wrapper/ocl_wrapper.h"
#include "libs/csvl/csvl.h"

#define N_WORK_GROUPS 256
#define N_WORK_ITEMS_PER_WORK_GROUP 32

cl_event max_find(cl_kernel k, cl_command_queue q, cl_mem output_buffer, cl_mem input_buffer,
                  cl_int n_elements, cl_int n_work_items, cl_int n_work_groups, cl_event to_wait)
{
    const size_t gws[] = { n_work_groups*n_work_items };
    const size_t lws[] = { n_work_items };
    cl_event max_find_event;
    cl_int err;

    cl_uint i = 0;
    err = clSetKernelArg(k, i++, sizeof(output_buffer), &output_buffer);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(input_buffer), &input_buffer);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(cl_float) * lws[0], NULL);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(n_elements), &n_elements);
    ocl_check(err, "Can't set max_find arg", i-1);

    if(to_wait == NULL)
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, lws, 0, NULL, &max_find_event);
    else
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, lws, 1, &to_wait, &max_find_event);

    ocl_check(err, "[FAIL] Can't enqueue max_find kernel");

    err = clFinish(q);
    ocl_check(err, "[FAIL] Can't complete command queue");

    return max_find_event;
}

int main(){
    printf("--------------------------------------------------\n");
    printf("              PARALLEL NORMALIZATION              \n");
    printf("--------------------------------------------------\n");

    // Loading Data:
    int n_elements;
    char * csv_pathname = "../data/credit_card_fraud_PCA.csv";
    float * host_buffer = csvl_load_fcolumn(csv_pathname, 3, &n_elements);
    if(host_buffer == NULL) return -1;

    // Wrapped OpenCL boilerplate:
    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context c = create_context(p, d);
    cl_command_queue q = create_queue(c, d);
    cl_program prog = create_program("../src/kernels/kernels.ocl", c, d);

    cl_int err;

    // CREATING THE OPENCL KERNELS:
    cl_kernel max_find_kernel = clCreateKernel(prog, "max_find", &err);
    ocl_check(err, "[FAIL] Can't create the kernel max_find");

    // COPYING THE HOST BUFFER TO A DEVICE BUFFER:
    cl_mem device_buffer = NULL;
    const size_t db_memsize = n_elements * sizeof(float);
    cl_mem_flags db_flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY;

    device_buffer = clCreateBuffer(c, db_flags, db_memsize, host_buffer, &err);
    ocl_check(err, "[FAIL] Can't copy host_buffer to device_buffer");

    // CREATING THE SUPPORT BUFFER:
    cl_mem support_buffer = NULL;
    const size_t sb_memsize = N_WORK_GROUPS * sizeof(float);
    cl_mem_flags sb_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;

    support_buffer = clCreateBuffer(c, sb_flags, sb_memsize, NULL, &err);
    ocl_check(err, "[FAIL] Can't create the support_buffer");

    // FINDING THE MAXIMUM:
    cl_event max_find_event[2], read_event;

    // Reducing the original buffer to N_WORK_GROUPS elements:
    max_find_event[0] = max_find(max_find_kernel, q, support_buffer, device_buffer, n_elements, 
                                 N_WORK_ITEMS_PER_WORK_GROUP, N_WORK_GROUPS, NULL);

    // Reducing the support buffer of N_WORK_GROUPS elements to only one element:
    max_find_event[1] = max_find(max_find_kernel, q, support_buffer, support_buffer, N_WORK_GROUPS, 
                                 N_WORK_ITEMS_PER_WORK_GROUP, 1, max_find_event[0]);

    float max;
    err = clEnqueueReadBuffer(q, support_buffer, CL_TRUE, 0, sizeof(max), &max, 1, max_find_event+1, &read_event);
    ocl_check(err, "[FAIL] Can't read the max value");

    printf("The max value is %f\n", max);

    clReleaseMemObject(device_buffer);
    clReleaseMemObject(support_buffer);
    clReleaseKernel(max_find_kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(c);
}

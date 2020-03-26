/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    kernel_launchers.c
    Collection of functions that allow to launch the OpenCL project's kernels
*/

#include "./kernel_launchers.h"

cl_event launch_normalize(cl_kernel k, cl_command_queue q, cl_device_id d,
                          cl_mem buffer_to_normalize, cl_int n_elements,
                          cl_float max, cl_float min)
{
    cl_int err;
    cl_event normalize_event;

    // Getting the preferred gws multiple:
    size_t gws_preferred_multiple;
    err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof(gws_preferred_multiple), &gws_preferred_multiple, NULL);
    ocl_check(err, "[FAIL] Can't get preferred gws multiple");

    const size_t gws[] = { round_mul_up(n_elements, gws_preferred_multiple) };

    // Argument passing to the kernel:
    cl_uint i = 0;
    err = clSetKernelArg(k, i++, sizeof(buffer_to_normalize), &buffer_to_normalize);
    ocl_check(err, "Can't set normalize arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(n_elements), &n_elements);
    ocl_check(err, "Can't set normalize arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(max), &max);
    ocl_check(err, "Can't set normalize arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(min), &min);
    ocl_check(err, "Can't set normalize arg", i-1);

    err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, &normalize_event);
    ocl_check(err, "[FAIL] Can't enqueue normalize kernel");

    // Waiting for all work items to complete:
    err = clFinish(q);
    ocl_check(err, "[FAIL] Can't complete command queue - launch_normalize");

    return normalize_event;
}

cl_event launch_max_find(cl_kernel k, cl_command_queue q, cl_event to_wait,
                         cl_mem output_buffer, cl_mem input_buffer, cl_int n_elements,
                         cl_int n_work_items, cl_int n_work_groups)
{
    const size_t gws[] = { n_work_groups * n_work_items };
    const size_t lws[] = { n_work_items };

    cl_event max_find_event;
    cl_int err;

    // Argument passing to the kernel:
    cl_uint i = 0;
    err = clSetKernelArg(k, i++, sizeof(output_buffer), &output_buffer);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(input_buffer), &input_buffer);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(cl_float) * lws[0], NULL);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(n_elements), &n_elements);
    ocl_check(err, "Can't set max_find arg", i-1);

    // Waiting for the given event:
    if(to_wait == NULL)
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, lws, 0, NULL, &max_find_event);
    else
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, lws, 1, &to_wait, &max_find_event);

    ocl_check(err, "[FAIL] Can't enqueue max_find kernel");

    // Waiting for all work items to complete:
    err = clFinish(q);
    ocl_check(err, "[FAIL] Can't complete command queue - launch_max_find");

    return max_find_event;
}

cl_event launch_min_find(cl_kernel k, cl_command_queue q, cl_event to_wait,
                         cl_mem output_buffer, cl_mem input_buffer, cl_int n_elements,
                         cl_int n_work_items, cl_int n_work_groups)
{
    const size_t gws[] = { n_work_groups * n_work_items };
    const size_t lws[] = { n_work_items };
    cl_event min_find_event;
    cl_int err;

    // Argument passing to the kernel:
    cl_uint i = 0;
    err = clSetKernelArg(k, i++, sizeof(output_buffer), &output_buffer);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(input_buffer), &input_buffer);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(cl_float) * lws[0], NULL);
    ocl_check(err, "Can't set max_find arg", i-1);
    err = clSetKernelArg(k, i++, sizeof(n_elements), &n_elements);
    ocl_check(err, "Can't set max_find arg", i-1);

    // Waiting for the given event:
    if(to_wait == NULL)
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, lws, 0, NULL, &min_find_event);
    else
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, lws, 1, &to_wait, &min_find_event);

    ocl_check(err, "[FAIL] Can't enqueue max_find kernel");

    // Waiting for all work items to complete:
    err = clFinish(q);
    ocl_check(err, "[FAIL] Can't complete command queue - launch_min_find");

    return min_find_event;
}

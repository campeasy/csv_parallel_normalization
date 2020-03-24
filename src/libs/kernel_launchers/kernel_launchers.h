/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    kernel_launchers.h
    Collection of functions that allow to launch the OpenCL project's kernels
*/

#pragma once

#include "../ocl_wrapper/ocl_wrapper.h"

#define MAX_FIND_KERNEL_NAME "max_find"
#define MIN_FIND_KERNEL_NAME "min_find"

cl_event launch_max_find(cl_kernel k, cl_command_queue q, cl_event to_wait,
                         cl_mem output_buffer, cl_mem input_buffer, cl_int n_elements,
                         cl_int n_work_items, cl_int n_work_groups);

cl_event launch_min_find(cl_kernel k, cl_command_queue q, cl_event to_wait,
                         cl_mem output_buffer, cl_mem input_buffer, cl_int n_elements,
                         cl_int n_work_items, cl_int n_work_groups);

/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    main.c
    CSV file parallel normalization using OpenCL
*/

#include "libs/csvl/csvl.h"
#include "libs/kernel_launchers/kernel_launchers.h"

#define N_WORK_GROUPS 256
#define N_WORK_ITEMS_PER_WORK_GROUP 32

float * normalize(float * host_buffer, int host_buffer_elements, float max, float min, int log,
                  cl_program ocl_program, cl_context ocl_context, cl_command_queue ocl_queue, cl_device_id ocl_device)
{
    cl_int err;
    cl_event normalize_event, read_event;
    float * normalized_buffer = malloc(sizeof(float) * host_buffer_elements);

    // Creating the OpenCL kernel:
    cl_kernel temp_k = clCreateKernel(ocl_program, NORMALIZE_KERNEL_NAME, &err);
    ocl_check(err, "[FAIL] Can't create the kernel ", NORMALIZE_KERNEL_NAME);

    // Creating the device buffer from the host buffer:
    cl_mem device_buffer = NULL;
    const size_t db_memsize = host_buffer_elements * sizeof(float);
    cl_mem_flags db_flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY;

    device_buffer = clCreateBuffer(ocl_context, db_flags, db_memsize, host_buffer, &err);
    ocl_check(err, "[FAIL] Can't create the device buffer - normalize");

    // Normalizing the device buffer:
    normalize_event = launch_normalize(temp_k, ocl_queue, ocl_device, device_buffer, host_buffer_elements, max, min);

    // Reading data from device:
    err = clEnqueueReadBuffer(ocl_queue, device_buffer, CL_TRUE, 0, db_memsize, normalized_buffer, 1, &normalize_event, &read_event);
    ocl_check(err, "[FAIL] Can't read the normalized buffer from device");

    if(log == 1){
        // Times and bandwidths check:
        const double normalize_ms = runtime_ms(normalize_event);
        const double normalize_gbs = (host_buffer_elements * sizeof(float) * 2)/1.0e6/normalize_ms;

        fprintf(stdout, "[LOG] Normalize:   %d elements, %.5f ms, %.5f GB/s\n", host_buffer_elements, normalize_ms, normalize_gbs);
    }

    clReleaseMemObject(device_buffer);
    clReleaseKernel(temp_k);

    return normalized_buffer;
}

float get_max(float * host_buffer, int host_buffer_elements, int log,
              cl_program ocl_program, cl_context ocl_context, cl_command_queue ocl_queue){
    cl_int err;
    cl_event max_find_event[2], read_event;
    float temp_max;

    // Creating the OpenCL kernel:
    cl_kernel temp_k = clCreateKernel(ocl_program, MAX_FIND_KERNEL_NAME, &err);
    ocl_check(err, "[FAIL] Can't create the kernel ", MAX_FIND_KERNEL_NAME);

    // Copying the host buffer to a device buffer:
    cl_mem device_buffer = NULL;
    const size_t db_memsize = host_buffer_elements * sizeof(float);
    cl_mem_flags db_flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY;

    device_buffer = clCreateBuffer(ocl_context, db_flags, db_memsize, host_buffer, &err);
    ocl_check(err, "[FAIL] Can't copy host buffer to device buffer - getting max");

    // Creating the support buffer:
    cl_mem support_buffer = NULL;
    const size_t sb_memsize = N_WORK_GROUPS * sizeof(float);
    cl_mem_flags sb_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;

    support_buffer = clCreateBuffer(ocl_context, sb_flags, sb_memsize, NULL, &err);
    ocl_check(err, "[FAIL] Can't create the support_buffer - getting max");

    // Reducing the original device buffer to N_WORK_GROUPS elements:
    max_find_event[0] = launch_max_find(temp_k, ocl_queue, NULL,
                                        support_buffer, device_buffer, host_buffer_elements, 
                                        N_WORK_ITEMS_PER_WORK_GROUP, N_WORK_GROUPS);

    // Reducing the support buffer of N_WORK_GROUPS elements to only one element:
    max_find_event[1] = launch_max_find(temp_k, ocl_queue, max_find_event[0],
                                        support_buffer, support_buffer, N_WORK_GROUPS, 
                                        N_WORK_ITEMS_PER_WORK_GROUP, 1);

    // Reading data from device:
    err = clEnqueueReadBuffer(ocl_queue, support_buffer, CL_TRUE, 0, sizeof(temp_max), &temp_max, 1, max_find_event+1, &read_event);
    ocl_check(err, "[FAIL] Can't read the max value from device");

    if(log == 1){
        // Times and bandwidths check:
        const double first_step_ms = runtime_ms(max_find_event[0]);
        const double first_step_gbs = (host_buffer_elements * sizeof(float) + N_WORK_GROUPS * sizeof(float))/1.0e6/first_step_ms;

        const double second_step_ms = runtime_ms(max_find_event[1]);
        const double second_step_gbs = (N_WORK_GROUPS * sizeof(float) + sizeof(float))/1.0e6/second_step_ms;

        const double total_ms = total_runtime_ms(max_find_event[0], max_find_event[1]);
        const double total_gbs = (first_step_gbs + second_step_gbs) / 2;

        fprintf(stdout, "[LOG] Getting Max: %d elements, %.5f ms, %.5f GB/s || Max: %f || Reduce 0: %.5f ms, %.5f GB/s - Reduce 1: %.5f ms, %.5f GB/s\n",
                host_buffer_elements, total_ms, total_gbs, temp_max, first_step_ms, first_step_gbs, second_step_ms, second_step_gbs);
    }

    clReleaseMemObject(device_buffer);
    clReleaseMemObject(support_buffer);
    clReleaseKernel(temp_k);

    return temp_max;
}

float get_min(float * host_buffer, int host_buffer_elements, int log,
              cl_program ocl_program, cl_context ocl_context, cl_command_queue ocl_queue){
    cl_int err;
    cl_event min_find_event[2], read_event;
    float temp_min;

    // Creating the OpenCL kernel:
    cl_kernel temp_k = clCreateKernel(ocl_program, MIN_FIND_KERNEL_NAME, &err);
    ocl_check(err, "[FAIL] Can't create the kernel ", MIN_FIND_KERNEL_NAME);

    // Copying the host buffer to a device buffer:
    cl_mem device_buffer = NULL;
    const size_t db_memsize = host_buffer_elements * sizeof(float);
    cl_mem_flags db_flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY;

    device_buffer = clCreateBuffer(ocl_context, db_flags, db_memsize, host_buffer, &err);
    ocl_check(err, "[FAIL] Can't copy host buffer to device buffer - getting min");

    // Creating the support buffer:
    cl_mem support_buffer = NULL;
    const size_t sb_memsize = N_WORK_GROUPS * sizeof(float);
    cl_mem_flags sb_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;

    support_buffer = clCreateBuffer(ocl_context, sb_flags, sb_memsize, NULL, &err);
    ocl_check(err, "[FAIL] Can't create the support_buffer - getting min");

    // Reducing the original device buffer to N_WORK_GROUPS elements:
    min_find_event[0] = launch_min_find(temp_k, ocl_queue, NULL,
                                        support_buffer, device_buffer, host_buffer_elements, 
                                        N_WORK_ITEMS_PER_WORK_GROUP, N_WORK_GROUPS);

    // Reducing the support buffer of N_WORK_GROUPS elements to only one element:
    min_find_event[1] = launch_min_find(temp_k, ocl_queue, min_find_event[0],
                                        support_buffer, support_buffer, N_WORK_GROUPS, 
                                        N_WORK_ITEMS_PER_WORK_GROUP, 1);

    // Reading data from device:
    err = clEnqueueReadBuffer(ocl_queue, support_buffer, CL_TRUE, 0, sizeof(temp_min), &temp_min, 1, min_find_event+1, &read_event);
    ocl_check(err, "[FAIL] Can't read the min value from device");

    if(log == 1){
        // Times and bandwidths check:
        const double first_step_ms = runtime_ms(min_find_event[0]);
        const double first_step_gbs = (host_buffer_elements * sizeof(float) + N_WORK_GROUPS * sizeof(float))/1.0e6/first_step_ms;

        const double second_step_ms = runtime_ms(min_find_event[1]);
        const double second_step_gbs = (N_WORK_GROUPS * sizeof(float) + sizeof(float))/1.0e6/second_step_ms;

        const double total_ms = total_runtime_ms(min_find_event[0], min_find_event[1]);
        const double total_gbs = (first_step_gbs + second_step_gbs) / 2;

        fprintf(stdout, "[LOG] Getting Min: %d elements, %.5f ms, %.5f GB/s || Min: %f || Reduce 0: %.5f ms, %.5f GB/s - Reduce 1: %.5f ms, %.5f GB/s\n",
                host_buffer_elements, total_ms, total_gbs, temp_min, first_step_ms, first_step_gbs, second_step_ms, second_step_gbs);
    }

    clReleaseMemObject(device_buffer);
    clReleaseMemObject(support_buffer);
    clReleaseKernel(temp_k);

    return temp_min;
}

int main(int argc, char *argv[]){
    printf("--------------------------------------------------\n");
    printf("              PARALLEL NORMALIZATION              \n");
    printf("--------------------------------------------------\n");

    if(argc < 3){
        fprintf(stdout, "[FAIL] Example of use: %s csv_pathname_to_normalize col_index1 col_index2 ... col_indexN \n", argv[0]);
        fprintf(stdout, "                       %s csv_pathname_to_normalize ALL\n", argv[0]);
        return -1;
    }

    int err;

    // Building the pathname:
    char * suffix = "../";
    char * temp_pathname = argv[1];

    char * csv_pathname = malloc(strlen(suffix) + strlen(temp_pathname) + 1);
    strcpy(csv_pathname, suffix);
    strcat(csv_pathname, temp_pathname);
    csv_pathname = realpath(csv_pathname, NULL);

    // Consistency Check:
    FILE * fd = fopen(csv_pathname, "r"); 
    if(fd == NULL){
        fprintf(stdout, "[FAIL] Given file does not exist\n");
        return -1;
    }
    fclose(fd);

    // Creating the array with the columns to normalize:
    int * cols_array;
    int cols_array_dim;

    if(strcmp("ALL", argv[2]) == 0){
        cols_array_dim = csvl_ncols(csv_pathname);
        cols_array_dim -= 1;
        cols_array = (int *) malloc(sizeof(int) * cols_array_dim);

        for(int i = 0; i<cols_array_dim; ++i){
            cols_array[i] = i+1;
        }
    }
    else{
        cols_array_dim = argc - 2;
        cols_array = (int *) malloc(sizeof(int) * cols_array_dim);

        int current = 0;
        for(int i = 0; i<cols_array_dim; ++i){
            cols_array[i] = atoi(argv[i+2]);
        }
    }

    // Wrapped OpenCL boilerplate:
    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context c = create_context(p, d);
    cl_command_queue q = create_queue(c, d);
    cl_program prog = create_program("../src/kernels/kernels.ocl", c, d);

    int n_elements;
    float * host_buffer;
    float temp_max, temp_min;

    fprintf(stdout, "[LOG] START normalization of %s\n", csv_pathname);

    for(int i=0; i<cols_array_dim; ++i)
    {
        fprintf(stdout, "\n");
        host_buffer = csvl_load_fcolumn(csv_pathname, cols_array[i], &n_elements);
        if(host_buffer == NULL) return -1;

        temp_max = get_max(host_buffer, n_elements, 1, prog, c, q);
        temp_min = get_min(host_buffer, n_elements, 1, prog, c, q);
        host_buffer = normalize(host_buffer, n_elements, temp_max, temp_min, 1, prog, c, q, d);

        fprintf(stdout, "[LOG] Writing changes to disk ...\n");
        err = csvl_write_fcolumn(csv_pathname, host_buffer, n_elements, cols_array[i]);
        if(err == -1) return -1;
    }

    fprintf(stdout, "\n[LOG] END normalization of %s\n", csv_pathname);

    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(c);
}

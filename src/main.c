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

    err = clEnqueueReadBuffer(ocl_queue, support_buffer, CL_TRUE, 0, sizeof(temp_max), &temp_max, 1, max_find_event+1, &read_event);
    ocl_check(err, "[FAIL] Can't read the max value from device");

    if(log == 1){
        // Times check:
        const double first_step_ms = runtime_ms(max_find_event[0]);
        const double first_step_gbs = (host_buffer_elements * sizeof(float) + N_WORK_GROUPS * sizeof(float))/1.0e6/first_step_ms;

        const double second_step_ms = runtime_ms(max_find_event[1]);
        const double second_step_gbs = (N_WORK_GROUPS * sizeof(float) + sizeof(float))/1.0e6/second_step_ms;

        const double total_ms = total_runtime_ms(max_find_event[0], max_find_event[1]);
        const double total_gbs = (first_step_gbs + second_step_gbs) / 2;

        fprintf(stdout, "[LOG] Getting Max: %g ms, %g GB/s  ||  Reduce 0: %g ms, %g GB/s - Reduce 1: %g ms, %g GB/s\n",
                total_ms, total_gbs, first_step_ms, first_step_gbs, second_step_ms, second_step_gbs);
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

    err = clEnqueueReadBuffer(ocl_queue, support_buffer, CL_TRUE, 0, sizeof(temp_min), &temp_min, 1, min_find_event+1, &read_event);
    ocl_check(err, "[FAIL] Can't read the min value from device");

    if(log == 1){
        // Times check:
        const double first_step_ms = runtime_ms(min_find_event[0]);
        const double first_step_gbs = (host_buffer_elements * sizeof(float) + N_WORK_GROUPS * sizeof(float))/1.0e6/first_step_ms;

        const double second_step_ms = runtime_ms(min_find_event[1]);
        const double second_step_gbs = (N_WORK_GROUPS * sizeof(float) + sizeof(float))/1.0e6/second_step_ms;

        const double total_ms = total_runtime_ms(min_find_event[0], min_find_event[1]);
        const double total_gbs = (first_step_gbs + second_step_gbs) / 2;

        fprintf(stdout, "[LOG] Getting Min: %g ms, %g GB/s  ||  Reduce 0: %g ms, %g GB/s - Reduce 1: %g ms, %g GB/s\n",
                total_ms, total_gbs, first_step_ms, first_step_gbs, second_step_ms, second_step_gbs);
    }

    clReleaseMemObject(device_buffer);
    clReleaseMemObject(support_buffer);
    clReleaseKernel(temp_k);

    return temp_min;
}

int main(){
    printf("--------------------------------------------------\n");
    printf("              PARALLEL NORMALIZATION              \n");
    printf("--------------------------------------------------\n");

    // Wrapped OpenCL boilerplate:
    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context c = create_context(p, d);
    cl_command_queue q = create_queue(c, d);
    cl_program prog = create_program("../src/kernels/kernels.ocl", c, d);

    char * csv_pathname = "../data/credit_card_fraud_PCA.csv";
    csv_pathname = realpath(csv_pathname, NULL);

    float * host_buffer;
    int n_elements;
    float temp_max, temp_min;

    int n_columns = csvl_ncols(csv_pathname);
    for(int i=1; i<n_columns; ++i){

        // Variable 'n_elements' will be filled with the i column dimension:
        host_buffer = csvl_load_fcolumn(csv_pathname, i, &n_elements);
        if(host_buffer == NULL) return -1;

        temp_max = get_max(host_buffer, n_elements, 0, prog, c, q);
        temp_min = get_min(host_buffer, n_elements, 0, prog, c, q);
        fprintf(stdout, "[LOG] Column %d : MAX -> %f, MIN -> %f\n\n", i, temp_max, temp_min);
    }

    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(c);
}

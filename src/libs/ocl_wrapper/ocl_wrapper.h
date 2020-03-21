/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    ocl_wrapper.h
    Collection of OpenCL C functions for wrapping the most
    common boilerplate of an OpenCL program
*/

// Including this file, the boilerplate can be reduced to:
# if 0
#include "./ocl_wrapper.h"

int main(int argc, char *argv[]){
    // force_platform("0");
    // force_device("1");

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("kernels.ocl", ctx, d);

    // Here starts the custom part: extract kernels,
    // allocate buffers, run kernels, get results, etc.

    return 0;
}
#endif

/*
    Include the headers defining the OpenCL host API
*/
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 120
#define BUFSIZE 4096

/*
    Fill the buffer 'buff_to_fill' with the content of
    the file specified in 'file_pathname'
*/
cl_int fill_buff(char * buff_to_fill, const char * file_pathname){
    FILE * file = fopen(file_pathname, "r");
    if(file == NULL){
        return -1;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    fread(buff_to_fill, 1, file_size, file);
    buff_to_fill[file_size] = '\0';
    fclose(file);

    return 0;
}

/*
    Check an OpenCL status, printing the messagge if it is an error
    and exiting in case of failure
*/
void ocl_check(cl_int err, const char *msg, ...){
    if (err != CL_SUCCESS){
        char msg_buf[BUFSIZE + 1];
        va_list ap;
        va_start(ap, msg);
        vsnprintf(msg_buf, BUFSIZE, msg, ap);
        va_end(ap);
        msg_buf[BUFSIZE] = '\0';
        fprintf(stderr, "%s - error %d\n", msg_buf, err);
        exit(1);
    }
}

/*
    Set the OCL_PLATFORM environment variable to the value of 'p'
*/
cl_int force_platform(const char * p){
    int status = setenv("OCL_PLATFORM", p, 1);
    return status;
}

/*
    Set the OCL_DEVICE environment variable to the value of 'd'
*/
cl_int force_device(const char * d){
    int status = setenv("OCL_DEVICE", d, 1);
    return status;
}

/*
    Return the ID of the platform specified in the OCL_PLATFORM
    environment variable, or the first one if the environment
    variable is not specified
*/
cl_platform_id select_platform(){
    printf("\n=========== OpenCL Wrapper v1.1 ===========\n");

    cl_uint nplats;
    cl_int err;
    cl_platform_id *plats;

    const char * const env = getenv("OCL_PLATFORM");
    cl_uint nump = 0;
    if (env && env[0] != '\0'){
        nump = atoi(env);
    }

    err = clGetPlatformIDs(0, NULL, &nplats);
    ocl_check(err, "[ERROR] counting platforms");

    printf("[OK] number of platforms: %u\n", nplats);

    plats = (cl_platform_id *) malloc(nplats * sizeof(*plats));
    err = clGetPlatformIDs(nplats, plats, NULL);
    ocl_check(err, "[ERROR] getting platform IDs");

    if (nump >= nplats){
        fprintf(stderr, "[ERROR] no platform number %u", nump);
        exit(1);
    }

    cl_platform_id choice = plats[nump];

    char buffer[BUFSIZE];
    err = clGetPlatformInfo(choice, CL_PLATFORM_NAME, BUFSIZE, buffer, NULL);
    ocl_check(err, "[ERROR] getting platform name");

    printf("[OK] selected platform:   %d\n", nump);
    printf("[OK] platform name:       %s\n", buffer);

    return choice;
}

/*
    Return the ID of the device of the given platform 'p' specified in the
    OCL_DEVICE environment variable or the first one if the environment
    variable is not specified
*/
cl_device_id select_device(cl_platform_id p){
    cl_uint ndevs;
    cl_int err;
    cl_device_id *devs;

    const char * const env = getenv("OCL_DEVICE");
    cl_uint numd = 0;
    if (env && env[0] != '\0'){
        numd = atoi(env);
    }

    err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevs);
    ocl_check(err, "[ERROR] counting devices");

    printf("[OK] number of devices:   %u\n", ndevs);

    devs = (cl_device_id *) malloc(ndevs * sizeof(*devs));
    err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndevs, devs, NULL);
    ocl_check(err, "devices #2");

    if(numd >= ndevs){
        fprintf(stderr, "[ERROR] no device number %u", numd);
        exit(1);
    }

    cl_device_id choice = devs[numd];
    char buffer[BUFSIZE];
    err = clGetDeviceInfo(choice, CL_DEVICE_NAME, BUFSIZE,buffer, NULL);
    ocl_check(err, "[ERROR] device name");

    printf("[OK] selected device:     %d\n", numd);
    printf("[OK] device name:         %s\n", buffer);

    return choice;
}

/*
    Create a one-device context
*/
cl_context create_context(cl_platform_id p, cl_device_id d){
    cl_int err;
    cl_context_properties ctx_prop[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0
    };

    cl_context ctx = clCreateContext(ctx_prop, 1, &d, NULL, NULL, &err);
    ocl_check(err, "[ERROR] create context");

    return ctx;
}

/*
    Create a command queue for the given device in the given context
*/
cl_command_queue create_queue(cl_context ctx, cl_device_id d){
    cl_int err;
    cl_command_queue que = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &err);
    ocl_check(err, "[ERROR] create queue");

    return que;
}

/*
    Compile the device part of the program, stored in the external
    file 'fname', for device 'dev' in context 'ctx'
*/
cl_program create_program(const char * const fname, cl_context ctx, cl_device_id dev){
    cl_int err, errlog;
    cl_program prg;

    char src_buf[BUFSIZE + 1];
    char *log_buf = NULL;
    size_t logsize;
    const char* buf_ptr = src_buf;
    time_t now = time(NULL);

    memset(src_buf, 0, BUFSIZE);
    err = fill_buff(src_buf, fname);
    if(err == -1){
        fprintf(stderr, "[ERROR] can't open file %s", fname);
        exit(1);
    }
    printf("\n[INFO] compiling kernels file: %s \n", fname);

    prg = clCreateProgramWithSource(ctx, 1, &buf_ptr, NULL, &err);
    ocl_check(err, "[ERROR] create program");

    err = clBuildProgram(prg, 1, &dev, "-I.", NULL, NULL);
    errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG,0, NULL, &logsize);
    ocl_check(errlog, "[ERROR] get program build log size");

    log_buf = (char *) malloc(logsize);
    errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, logsize, log_buf, NULL);
    ocl_check(errlog, "[ERROR] get program build log");

    while (logsize > 0 &&
            (log_buf[logsize-1] == '\n' ||
            log_buf[logsize-1] == '\0')){
        logsize--;
    }
    if (logsize > 0) {
        log_buf[logsize] = '\n';
        log_buf[logsize+1] = '\0';
    }
    else{
        log_buf[logsize] = '\0';
    }
    printf("\n--------- COMPILATION PROCESS LOG ---------\n%s", log_buf);
    ocl_check(err, "[ERROR] build program");

    printf("\n===========================================\n\n");
    return prg;
}

/*
    Runtime of an event, in nanoseconds.
    Note that if NS is the runtimen of an event in nanoseconds and NB is the number of
    byte read and written during the event, NB/NS is the effective bandwidth expressed in GB/s
*/
cl_ulong runtime_ns(cl_event evt){
    cl_int err;
    cl_ulong start, end;

    err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ocl_check(err, "[ERROR] get start");

    err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    ocl_check(err, "[ERROR] get end");

    return (end - start);
}

cl_ulong total_runtime_ns(cl_event from, cl_event to){
    cl_int err;
    cl_ulong start, end;

    err = clGetEventProfilingInfo(from, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ocl_check(err, "[ERROR] get start");
    err = clGetEventProfilingInfo(to, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    ocl_check(err, "[ERROR] get end");

    return (end - start);
}

/*
    Runtime of an event, in milliseconds:
*/
double runtime_ms(cl_event evt){
    return runtime_ns(evt) * 1.0e-6;
}

double total_runtime_ms(cl_event from, cl_event to){
    return total_runtime_ns(from, to)*1.0e-6;
}

/*
    Round gws to the next multiple of lws:
*/
size_t round_mul_up(size_t gws, size_t lws){
    return ((gws + lws - 1)/lws)*lws;
}

double bandwidth_gbps(int n, size_t memsize, double ms){
    return n*memsize/1.0e6/ms;
}

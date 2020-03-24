/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    ocl_wrapper.c
    Collection of OpenCL C functions for wrapping the most
    common boilerplate of an OpenCL program
*/

#include "./ocl_wrapper.h"

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

cl_int force_platform(const char * p){
    int status = setenv("OCL_PLATFORM", p, 1);
    return status;
}

cl_int force_device(const char * d){
    int status = setenv("OCL_DEVICE", d, 1);
    return status;
}

cl_platform_id select_platform(){
    printf("\n---------------- OpenCL Wrapper ------------------\n");

    cl_uint nplats;
    cl_int err;
    cl_platform_id *plats;

    const char * const env = getenv("OCL_PLATFORM");
    cl_uint nump = 0;
    if (env && env[0] != '\0'){
        nump = atoi(env);
    }

    err = clGetPlatformIDs(0, NULL, &nplats);
    ocl_check(err, "[ERROR] Counting platforms");

    printf("[OK] Number of platforms: %u\n", nplats);

    plats = (cl_platform_id *) malloc(nplats * sizeof(*plats));
    err = clGetPlatformIDs(nplats, plats, NULL);
    ocl_check(err, "[ERROR] Getting platform IDs");

    if (nump >= nplats){
        fprintf(stderr, "[ERROR] No platform number %u", nump);
        exit(1);
    }

    cl_platform_id choice = plats[nump];

    char buffer[BUFSIZE];
    err = clGetPlatformInfo(choice, CL_PLATFORM_NAME, BUFSIZE, buffer, NULL);
    ocl_check(err, "[ERROR] Getting platform name");

    printf("[OK] Selected platform:   %d\n", nump);
    printf("[OK] Platform name:       %s\n", buffer);

    return choice;
}

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
    ocl_check(err, "[ERROR] Counting devices");

    printf("[OK] Number of devices:   %u\n", ndevs);

    devs = (cl_device_id *) malloc(ndevs * sizeof(*devs));
    err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndevs, devs, NULL);
    ocl_check(err, "devices #2");

    if(numd >= ndevs){
        fprintf(stderr, "[ERROR] No device number %u", numd);
        exit(1);
    }

    cl_device_id choice = devs[numd];
    char buffer[BUFSIZE];
    err = clGetDeviceInfo(choice, CL_DEVICE_NAME, BUFSIZE,buffer, NULL);
    ocl_check(err, "[ERROR] Device name");

    printf("[OK] Selected device:     %d\n", numd);
    printf("[OK] Device name:         %s\n", buffer);

    return choice;
}

cl_context create_context(cl_platform_id p, cl_device_id d){
    cl_int err;
    cl_context_properties ctx_prop[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0
    };

    cl_context ctx = clCreateContext(ctx_prop, 1, &d, NULL, NULL, &err);
    ocl_check(err, "[ERROR] Create context");

    return ctx;
}

cl_command_queue create_queue(cl_context ctx, cl_device_id d){
    cl_int err;
    cl_command_queue que = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &err);
    ocl_check(err, "[ERROR] Create queue");

    return que;
}

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
        fprintf(stderr, "[ERROR] Can't open file %s", fname);
        exit(1);
    }
    printf("\n[OK] Compiling kernels file: %s", fname);

    prg = clCreateProgramWithSource(ctx, 1, &buf_ptr, NULL, &err);
    ocl_check(err, "[ERROR] Create program");

    err = clBuildProgram(prg, 1, &dev, "-I.", NULL, NULL);
    errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG,0, NULL, &logsize);
    ocl_check(errlog, "[ERROR] Get program build log size");

    log_buf = (char *) malloc(logsize);
    errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, logsize, log_buf, NULL);
    ocl_check(errlog, "[ERROR] Get program build log");

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

    if(err != CL_SUCCESS){
        printf("\n---------------- COMPILATION LOG ---------------\n%s", log_buf);
    }
    ocl_check(err, "[ERROR] Can't build program");
    printf("\n--------------------------------------------------\n\n");

    return prg;
}

cl_ulong runtime_ns(cl_event evt){
    cl_int err;
    cl_ulong start, end;

    err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ocl_check(err, "[ERROR] Get start");

    err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    ocl_check(err, "[ERROR] Get end");

    return (end - start);
}

cl_ulong total_runtime_ns(cl_event from, cl_event to){
    cl_int err;
    cl_ulong start, end;

    err = clGetEventProfilingInfo(from, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ocl_check(err, "[ERROR] Get start");
    err = clGetEventProfilingInfo(to, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    ocl_check(err, "[ERROR] Get end");

    return (end - start);
}

double runtime_ms(cl_event evt){
    return runtime_ns(evt) * 1.0e-6;
}

double total_runtime_ms(cl_event from, cl_event to){
    return total_runtime_ns(from, to)*1.0e-6;
}

size_t round_mul_up(size_t gws, size_t lws){
    return ((gws + lws - 1)/lws)*lws;
}

double bandwidth_gbps(int n, size_t memsize, double ms){
    return n*memsize/1.0e6/ms;
}

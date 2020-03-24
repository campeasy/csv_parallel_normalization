/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    ocl_wrapper.h
    Collection of OpenCL C functions for wrapping the most
    common boilerplate of an OpenCL program
*/

#pragma once

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
#define BUFSIZE 16384

/*
    Fill the buffer 'buff_to_fill' with the content of
    the file specified in 'file_pathname'
*/
cl_int fill_buff(char * buff_to_fill, const char * file_pathname);

/*
    Check an OpenCL status, printing the messagge if it is an error
    and exiting in case of failure
*/
void ocl_check(cl_int err, const char *msg, ...);

/*
    Set the OCL_PLATFORM environment variable to the value of 'p'
*/
cl_int force_platform(const char * p);

/*
    Set the OCL_DEVICE environment variable to the value of 'd'
*/
cl_int force_device(const char * d);

/*
    Return the ID of the platform specified in the OCL_PLATFORM
    environment variable, or the first one if the environment
    variable is not specified
*/
cl_platform_id select_platform();

/*
    Return the ID of the device of the given platform 'p' specified in the
    OCL_DEVICE environment variable or the first one if the environment
    variable is not specified
*/
cl_device_id select_device(cl_platform_id p);

/*
    Create a one-device context
*/
cl_context create_context(cl_platform_id p, cl_device_id d);

/*
    Create a command queue for the given device in the given context
*/
cl_command_queue create_queue(cl_context ctx, cl_device_id d);

/*
    Compile the device part of the program, stored in the external
    file 'fname', for device 'dev' in context 'ctx'
*/
cl_program create_program(const char * const fname, cl_context ctx, cl_device_id dev);

/*
    Runtime of an event, in nanoseconds.
    Note that if NS is the runtimen of an event in nanoseconds and NB is the number of
    byte read and written during the event, NB/NS is the effective bandwidth expressed in GB/s
*/
cl_ulong runtime_ns(cl_event evt);

cl_ulong total_runtime_ns(cl_event from, cl_event to);

/*
    Runtime of an event, in milliseconds:
*/
double runtime_ms(cl_event evt);

double total_runtime_ms(cl_event from, cl_event to);

/*
    Round gws to the next multiple of lws:
*/
size_t round_mul_up(size_t gws, size_t lws);

double bandwidth_gbps(int n, size_t memsize, double ms);

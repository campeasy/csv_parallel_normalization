/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    csvl.h
    C library for processing a CSV file type and speeding up its elaboration
*/

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define KB 1024
#define MB 1024 * KB

#define ROW_MAX_SIZE KB

/*
    This routine takes the pathname of a CSV file and returns
    its number of rows or -1 if something goes wrong.
*/
int csvl_nrows (const char * csv_path);

/*
    This routine takes the pathname of a CSV file and returns
    its number of columns or -1 if something goes wrong.
*/
int csvl_ncols (const char * csv_path);

/*
    This routine takes the pathname of a CSV file and print
    the content of the CSV file on the standard output.
*/
void csvl_print (const char * csv_pathname);

/*
    This routine takes the pathname of a CSV file and the pathname of a new CSV file.
    The new CSV file will be created with the content of the columns specified in the given array.
    The routine returns 0 if everything is OK, -1 instead.
*/
int csvl_columns_to_file (const char * csv_original_path,
                          const char* csv_new_path,
                          const int * columns_array,
                          const int columns_array_dim);

/*
    This routine takes the pathname of a CSV file and the pathname of a new CSV file.
    The new CSV file will be a one-column CSV, with the content of the column specified.
    The routine returns 0 if everything is OK, -1 instead.
*/
int csvl_column_to_file (const char * csv_original_path,
                         const char * csv_new_path,
                         const int column_number);

/*
    This routine takes the pathname of a CSV file and load a specified FLOAT column
    in a buffer that will be returned.
    The routine willl also fill a pointer with the dimension of the returned buffer.
    The routine returns NULL if fails, or the pointer to the data if success.
*/
float * csvl_load_fcolumn(const char * csv_pathname,
                          const int column_number,
                          int * buffer_dim);

/*
    This routine takes the pathname of a CSV file and replace a specified column
    with a FLOAT given buffer.
    The routine returns 0 if everything is OK, -1 instead.
*/
int csvl_write_fcolumn(const char * csv_path,
                       const float * buffer_to_write,
                       const int buffer_dim,
                       const int column_number_to_ovverride);

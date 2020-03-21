/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    csvl_test.c
    C program for testing the CSVL library.
*/

#include "../libs/csvl/csvl.h"

// Data for Testing:

int   N_ROWS_CSV_TEST      = 5;
int   N_COLS_CSV_TEST      = 3;
int   SAMPLE_BUFFER_DIM    = 4;
int   FLOAT_ARRAY_TEST_DIM = 4;

float FLOAT_ARRAY_TEST[]   = {3.100,4.200,5.300,6.400};
float SAMPLE_BUFFER[]      = {1.100,2.200,3.300,4.400};
char  csv_test_pathname[]  = "../../data/csvl_test.csv";

// Routines for Testing:

int test_ncols(){
    int n_cols = csvl_ncols(csv_test_pathname);
    if(n_cols != N_COLS_CSV_TEST){
        fprintf(stderr, "[CSVL TEST][FAIL] Error counting the number of cols of the CSV file\n");
        return -1;
    }
    fprintf(stdout, "[CSVL TEST][OK] Corrrectly counted the number of cols of the CSV file \n");
    return n_cols;
}

int test_nrows(){
    int n_rows = csvl_nrows(csv_test_pathname);
    if(n_rows != N_ROWS_CSV_TEST){
        fprintf(stderr, "[CSVL TEST][FAIL] Error counting the number of rows of the CSV file\n");
        return -1;
    }
    fprintf(stdout, "[CSVL TEST][OK] Corrrectly counted the number of rows of the CSV file \n");
    return n_rows;
}

void test_load_fcolumn(){
    int n_rows = test_nrows();
    int n_elements;
    float * float_column_buffer = csvl_load_fcolumn(csv_test_pathname, 2, &n_elements);
    if(n_elements != n_rows - 1){
        fprintf(stderr, "[CSVL TEST][FAIL] Error loading a float column of the CSV file\n");
        return;
    }
    for(int i = 0; i < n_elements; ++i){
        if(float_column_buffer[i] != FLOAT_ARRAY_TEST[i]){
            fprintf(stderr, "[CSVL TEST][FAIL] Error loading a float column of the CSV file\n");
            return;
        }
    }
    fprintf(stdout, "[CSVL TEST][OK] Corrrectly loaded a float column of the CSV file\n");
    return;
}

void test_write_fcolumn(){
    csvl_write_fcolumn(csv_test_pathname, SAMPLE_BUFFER, SAMPLE_BUFFER_DIM, 2);

    int n_rows = test_nrows();
    int n_elements;
    float * float_column_buffer = csvl_load_fcolumn(csv_test_pathname, 2, &n_elements);
    if(n_elements != n_rows - 1){
        fprintf(stderr, "[CSVL TEST][FAIL] Error loading an ovverode float column of the CSV file\n");
        return;
    }
    for(int i = 0; i < n_elements; ++i){
        if(float_column_buffer[i] != SAMPLE_BUFFER[i]){
            fprintf(stderr, "[CSVL TEST][FAIL] Error loading an overrode float column of the CSV file\n");
            return;
        }
    }

    fprintf(stdout, "[CSVL TEST][OK] Corrrectly override a float column of the CSV file\n");
    csvl_write_fcolumn(csv_test_pathname, FLOAT_ARRAY_TEST, FLOAT_ARRAY_TEST_DIM, 2);
    return;
}

void test_print(){
    csvl_print(csv_test_pathname);
}

// Program for Testing:

int main(int argc, char * argv[]){
    fprintf(stdout, "--------------------------------------------------\n");
    fprintf(stdout, " CSVL TEST - C library for processing a CSV file\n");
    fprintf(stdout, "--------------------------------------------------\n");

    // TESTING NUMBER OF COLUMNS:
    test_ncols();

    // TESTING NUMBER OF ROWS:
    test_nrows();

    // TESTING THE LOAD OF A FLOAT COLUMN:
    test_load_fcolumn();

    // TESTING THE WRITING OF A FLOAT COLUMN:
    test_write_fcolumn();

    // TESTING THE PRINTING OF THE FILE:
    // test_print();

    return 0;
}

/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    csvl_test.c
    C program which use the CSVL library for filtering a CSV file
*/

#include "../libs/csvl/csvl.h"

int main(int argc, char * argv[]){
    if(argc < 3){
        fprintf(stdout, "[CSVL FILTER][FAIL] Filter example of use: %s csv_pathname csv_new_pathname col_index1 col_index2 ... col_indexN \n", argv[0]);
        return -1;
    }

    char * original_path = argv[1];
    char * new_path = argv[2];

    // Creating the array with the new columns:
    int dim = argc - 3;
    int * indexes = (int *) malloc(sizeof(int) * dim);
    for(int i = 0; i < dim; ++i){
        indexes[i] = atoi(argv[i+3]);
    }

    // Filtering the original CSV file:
    if(csvl_columns_to_file(original_path, new_path, indexes, dim) != 0){
        fprintf(stdout, "[CSVL FILTER][FAIL] Can't filter CSV file\n");
        return -1;
    }
    else fprintf(stdout, "[CSVL FILTER][OK] CSV filtered successfully\n");

    return 0;
}

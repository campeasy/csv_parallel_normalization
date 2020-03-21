/*
    AY 19/20
    Salvatore Campisi
    Parallel Programming on GPU
    CSV Parallel Normalization

    csvl.c
    C library for processing a CSV file type and speeding up its elaboration.
*/

#include "csvl.h"

int csvl_nrows(const char * csv_path)
{
    // Opening the CSV file:
    FILE * csv_fd = fopen(csv_path, "r");
    if(csv_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Can't read %s\n", csv_path);
        return -1;
    }

    int rows_counter = 0;
    char temp_row[ROW_MAX_SIZE];

    // Rows counting:
    while(fgets(temp_row, ROW_MAX_SIZE, csv_fd) != NULL){
        ++rows_counter;
    }

    fclose(csv_fd);
    return rows_counter;
}

int csvl_ncols(const char * csv_path)
{
    // Opening the CSV file:
    FILE * csv_fd = fopen(csv_path, "r");
    if(csv_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Can't read %s\n", csv_path);
        return -1;
    }

    int cols_counter = 0;
    char temp_row[ROW_MAX_SIZE];
    const char * sep = ",";
    char * temp_piece;

    // Getting the file's first row:
    fgets(temp_row, ROW_MAX_SIZE, csv_fd);

    // Counting the columns:
    temp_piece = strtok(temp_row, sep);
    while(temp_piece != NULL){
        ++cols_counter;
        temp_piece = strtok(NULL, sep);
    }

    fclose(csv_fd);
    return cols_counter;
}

void csvl_print(const char * csv_path)
{
    // Opening the CSV file:
    FILE * csv_fd = fopen(csv_path, "r");
    if(csv_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Can't read %s\n", csv_path);
        return;
    }

    char temp_row[ROW_MAX_SIZE];

    // Printing the CSV file:
    while(fgets(temp_row, ROW_MAX_SIZE, csv_fd) != NULL){
        fprintf(stdout, "%s", temp_row);
    }

    fclose(csv_fd);
    return;
}

int csvl_columns_to_file(const char * csv_original_path,
                         const char * csv_new_path,
                         const int * columns_array,
                         const int columns_array_dim)
{
    // Opening the CSV file:
    FILE * csv_original_fd = fopen(csv_original_path, "r");
    if(csv_original_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Can't read %s\n", csv_original_path);
        return -1;
    }

    // Creating the new CSV file:
    FILE * csv_new_fd = fopen(csv_new_path, "w+");
    if(csv_new_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Can't create %s\n", csv_new_path);
        return -1;
    }

    char temp_row[ROW_MAX_SIZE];
    const char * sep = ",";
    char * temp_piece;
    int current_column_index = 0;
    int p = 0;

    // For each row in the CSV file:
    while(fgets(temp_row, ROW_MAX_SIZE, csv_original_fd) != NULL){
        current_column_index = 0;
        temp_piece = strtok(temp_row, sep);

        // For each piece of the current row:
        while(temp_piece != NULL){
            ++current_column_index;

            // For each column to write in the new file:
            for(int i = 0; i < columns_array_dim; ++i){

                //Check if this column is one of the columns that must be written in the new file:
                if(current_column_index == columns_array[i]){

                    // If this is the last column, remove the '\n':
                    if(i == columns_array_dim - 1){
                        p = strlen(temp_piece) - 1;
                        if(temp_piece[p] == '\n') temp_piece[p] = '\0';
                    }

                    // Write the piece in the new file:
                    fprintf(csv_new_fd, "%s", temp_piece);

                    // If this is not the last column, insert the separator:
                    if(i != columns_array_dim - 1) fprintf(csv_new_fd, ",");
                }

            }
            temp_piece = strtok(NULL, sep);
        }
        fprintf(csv_new_fd, "\n");
    }

    fprintf(stdout, "[CSVL - OK] %s correctly processed\n", csv_original_path);
    fprintf(stdout, "[CSVL - OK] %s correctly built\n", csv_new_path);
    fclose(csv_new_fd);
    fclose(csv_original_fd);
    return 0;
}

int csvl_column_to_file(const char * csv_original_path,
                        const char * csv_new_path,
                        const int column_number)
{
    const int columns[] = {column_number};
    int result_code = csvl_columns_to_file(csv_original_path, csv_new_path, columns, 1);
    if(result_code != 0) return result_code;
    return 0;
}

float * csvl_load_fcolumn(const char * csv_path,
                          const int column_number,
                          int * buffer_dim)
{
    char * temp_path = "./temp.csv";

    // Creating the temporary CSV file:
    int result = csvl_column_to_file(csv_path, temp_path, column_number);
    if(result != 0) return NULL;

    // Opening the temporary CSV file:
    FILE * csv_fd = fopen(temp_path, "r");
    if(csv_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Can't process %s\n", csv_path);
        return NULL;
    }

    char temp_row[ROW_MAX_SIZE];

    // Allocating the array for the data:
    int data_dim = csvl_nrows(temp_path) - 1;
    float * csv_data = (float *) malloc(sizeof(float) * data_dim);

    // Skipping the first row of the CSV file (is the one with the column name):
    fgets(temp_row, ROW_MAX_SIZE, csv_fd);

    // Loading the specidied column into the buffer:
    int i = 0;
    while(fgets(temp_row, ROW_MAX_SIZE, csv_fd) != NULL){
        if(i < data_dim){
            csv_data[i] = atof(temp_row);
            ++i;
        }
    }

    // Removing the temporary CSV file:
    fclose(csv_fd);
    remove(temp_path);

    fprintf(stdout, "[CSVL - OK] Correctly loaded float column %d from %s\n", column_number, csv_path);

    // Return values:
    * buffer_dim = data_dim;
    return csv_data;
}

int csvl_write_fcolumn(const char * csv_path,
                       const float * buffer_to_write,
                       const int buffer_dim,
                       const int column_number_to_ovverride)
{
    // Checking if the CSV file already exist:
    FILE * csv_fd = fopen(csv_path, "r");
    if(csv_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] File %s does not exist\n", csv_path);
        return -1;
    }

    int csv_file_ncols = csvl_ncols(csv_path);

    // Consistency Checks:
    if(buffer_to_write == NULL || buffer_dim == 0){
        fprintf(stderr, "[CSVL - FAIL] Error processing %s, the given buffer is not valid\n", csv_path);
        return -1;
    }
    if(buffer_dim > (csvl_nrows(csv_path) - 1)){
        fprintf(stderr, "[CSVL - FAIL] Error processing %s, the given buffer is not valid\n", csv_path);
        return -1;
    }
    if(column_number_to_ovverride < 1 || column_number_to_ovverride > csv_file_ncols){
        fprintf(stderr, "[CSVL - FAIL] Error processing %s, the selected column is not valid\n", csv_path);
        return -1;
    }

    // Creating the new CSV file:
    char * temp_path = "./temp.csv";

    FILE * temp_fd = fopen(temp_path, "w+");
    if(temp_fd == NULL){
        fprintf(stderr, "[CSVL - FAIL] Error can't write the changes %s\n", csv_path);
        return -1;
    }

    // Writing on the new CSV file:

    char temp_row[ROW_MAX_SIZE];
    char * temp_piece;
    const char * sep = ",";
    int current_column_index = 0;
    int row_counter = 0;

    // Skip the process of the first row, you have only to rewrite it: (column names)
    fgets(temp_row, ROW_MAX_SIZE, csv_fd);
    fprintf(temp_fd, "%s", temp_row);

    while(fgets(temp_row, ROW_MAX_SIZE, csv_fd) != NULL){
        current_column_index = 0;
        temp_piece = strtok(temp_row, sep);

        // For each piece of the current row:
        while(temp_piece != NULL){
            ++current_column_index;

            // If we must override this column:
            if(current_column_index == column_number_to_ovverride){
                // If this is the last column:
                if(current_column_index == csv_file_ncols){
                    fprintf(temp_fd, "%.3f\n", buffer_to_write[row_counter]);
                }
                else fprintf(temp_fd, "%.3f,", buffer_to_write[row_counter]);
            }

            // If this column must not be ovverriden:
            else{
                // If this is the last column:
                if(current_column_index == csv_file_ncols){
                    fprintf(temp_fd, "%s", temp_piece);
                }
                else fprintf(temp_fd, "%s,", temp_piece);
            }

            temp_piece = strtok(NULL, sep);
        }
        ++row_counter;
    }

    // Swapping the old CSV file with new CSV file:
    char * support_path = "./support.csv";
    rename(csv_path, support_path);

    if(rename(temp_path, csv_path) != 0){
        rename(support_path, csv_path);

        remove(temp_path);
        fprintf(stderr, "[CSVL - FAIL] Error can't complete the changes %s\n", csv_path);
        return -1;
    }

    remove(temp_path);
    remove(support_path);

    fclose(csv_fd);
    fclose(temp_fd);

    return 0;
}

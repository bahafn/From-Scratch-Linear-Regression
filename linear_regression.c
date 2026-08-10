#include "linear_regression.h"
#include "string_utils.h"
#include "dynamic_array.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

DataSet create_dataset(size_t samples, size_t features,
                       double (*feature_matrix)[features],
                       double *target_vector) {

    DataSet dataset;

    dataset.feature_matrix = create_matrix(samples, features, feature_matrix);
    dataset.target_vector  = target_vector;

    return dataset;
}

void init_dataset(DataSet *dataset, size_t samples, size_t features) {
    dataset->feature_matrix = create_empty_matrix(samples, features);
    dataset->target_vector  = malloc(samples * sizeof(*dataset->target_vector));
}

void destroy_dataset(DataSet *dataset) {
    if (dataset) {
        destroy_matrix(&dataset->feature_matrix);
        free(dataset->target_vector);
        dataset->target_vector = NULL;
    }
}

#define MAX_LINE_LENGTH 65536

DataSet *read_csv_dataset(const char *file_path, size_t target_column_index) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", file_path);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END)) {
        fprintf(stderr, "Failed to read file: %s\n", file_path);
        fclose(file);
        return NULL;
    }

    size_t file_size = ftell(file);

    // Average number of double entries in a file of this size
    // The max number is the same as file_size (all entires empty, just commas)
    // The max number for a full file is file_size / 2 (one digit for each and a comma or endline)
    // We assume that the average entry would have about 3 digits
    size_t estimated_entries = file_size / 4;
    double_da data;
    da_init(data, estimated_entries);

    // Restart from the beginning of the file
    rewind(file);
    char line_buffer[MAX_LINE_LENGTH];

    size_t rows = 0;
    size_t cols = 0;

    // Used because we don't want to allocate the target vector until we estimate how many rows are there.
    double first_in_target_vector = 0;

    // Count number of cols from the first line
    if (fgets(line_buffer, sizeof(line_buffer), file)) {
        rows++;

        char *token = strtok(line_buffer, ",");
        while (token) {
            cols++;

            if (cols - 1 != target_column_index) {
                da_append(data, string_to_double(token));
            } else {
                first_in_target_vector = string_to_double(token);
            }

            token = strtok(NULL, ",");
        }
    }

    // Currently has the estimated number of rows depending on the estimated_entries and cols
    double_da target_vector;
    da_init(target_vector, estimated_entries / cols);
    da_append(target_vector, first_in_target_vector);

    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        char *token = strtok(line_buffer, ",");

        for (size_t col = 0; col < cols; col++) {
            if (col != target_column_index) {
                da_append(data, string_to_double(token));
            } else {
                da_append(target_vector, string_to_double(token));
            }

            token = strtok(NULL, ",");
        }

        rows++;
    }

    da_resize(data, rows * (cols - 1));
    da_resize(target_vector, rows);

    DataSet *dataset = malloc(sizeof(DataSet));
    dataset->target_vector       = target_vector.items;
    dataset->feature_matrix.rows = rows;
    dataset->feature_matrix.cols = cols - 1; // - 1 because we counted the target_vector
    dataset->feature_matrix.data = data.items;

    fclose(file);
    return dataset;
}

Min_Max_Scaler_Set min_max_fit(const DataSet *dataset,
                               size_t column_count,
                               const size_t columns[]) {

    const Matrix *feature_matrix = &dataset->feature_matrix;

    size_t rows = feature_matrix->rows;
    size_t cols = feature_matrix->cols;

    if (!columns) {
        column_count = cols;
    }

    Min_Max_Scaler_Set scaler_set;
    scaler_set.scaler_count = column_count;
    scaler_set.scalers      = malloc(column_count * sizeof(*scaler_set.scalers));

    if (!scaler_set.scalers) {
        scaler_set.scaler_count = 0;
        return scaler_set;
    }

    // All columns
    if (!columns) {
        // Loop over first row and all column scalers to init them
        for (size_t i = 0; i < cols; i++) {
            scaler_set.scalers[i].col_index = i;
            scaler_set.scalers[i].min_value = feature_matrix->data[i];
            scaler_set.scalers[i].max_value = feature_matrix->data[i];
        }

        for (size_t row = 1; row < rows; row++) {
            size_t base = row * cols;

            for (size_t col = 0; col < cols; col++) {
                double value = feature_matrix->data[base + col];

                Min_Max_Scaler *scaler = &scaler_set.scalers[col];

                if (value < scaler->min_value) {
                    scaler->min_value = value;
                } else if (value > scaler->max_value) {
                    scaler->max_value = value;
                }
            }
        }
    } else { // Selected columns
        for (size_t i = 0; i < column_count; i++) {
            size_t col = columns ? columns[i] : i;

            double col_max = -INFINITY;
            double col_min =  INFINITY;

            size_t index = col;
            for (size_t row = 0; row < rows; row++) {
                double value = feature_matrix->data[index];
                index += cols;

                if (value < col_min) {
                    col_min = value;
                } else if (value > col_max) {
                    col_max = value;
                }
            }

            scaler_set.scalers[i].min_value = col_min;
            scaler_set.scalers[i].max_value = col_max;
            scaler_set.scalers[i].col_index = col;
        }
    }

    return scaler_set;
}


Min_Max_Scaler_Set min_max_fit_all(const DataSet *dataset) {
    return min_max_fit(dataset, dataset->feature_matrix.cols, NULL);
}

void min_max_transform(DataSet *dataset, const Min_Max_Scaler_Set *scaler_set) {
    Matrix *matrix = &dataset->feature_matrix;

    size_t rows = matrix->rows;
    size_t cols = matrix->cols;

    // All columns
    if (scaler_set->scaler_count == matrix->cols) {
        for (size_t row = 0; row < rows; row++) {
            size_t base = row * cols;

            for (size_t col = 0; col < cols; col++) {
                const Min_Max_Scaler *scaler = &scaler_set->scalers[col];

                double range = scaler->max_value - scaler->min_value;

                if (range == 0) {
                    matrix->data[base + col] = 0;
                } else  {
                    matrix->data[base + col] = (matrix->data[base + col] - scaler->min_value) / range;
                }
            }
        }
    } else {
        for (size_t i = 0; i < scaler_set->scaler_count; i++) {
            const Min_Max_Scaler *scaler = &scaler_set->scalers[i];

            size_t col   = scaler->col_index;
            double range = scaler->max_value - scaler->min_value;
            size_t index = col;
            for (size_t row = 0; row < rows; row++) {
                if (range == 0) {
                    matrix->data[index] = 0;
                } else {
                    matrix->data[index] = (matrix->data[index] - scaler->min_value) / range;
                }

                index += cols;
            }
        }
    }
}

void min_max_transform_row(double *row, const Min_Max_Scaler_Set *scaler_set) {
    for (size_t i = 0; i < scaler_set->scaler_count; i++) {
        Min_Max_Scaler *scaler = &scaler_set->scalers[i];

        size_t col   = scaler->col_index;
        double range = scaler->max_value - scaler->min_value;

        if (scaler->min_value == scaler->max_value) {
            row[col] = 0;
        } else {
            row[col] = (row[col] - scaler->min_value) / range;
        }
    }
}

void destroy_scaler_set(Min_Max_Scaler_Set *scaler_set) {
    free(scaler_set->scalers);
}

static void build_normal_equation(const DataSet *dataset, Matrix *a, double *b) {
    size_t rows = dataset->feature_matrix.rows;
    size_t cols = dataset->feature_matrix.cols;

    size_t a_cols = cols + 1;

    const double *x = dataset->feature_matrix.data;
    const double *y = dataset->target_vector;

    for (size_t row = 0; row < rows; row++) {
        const double *sample = &x[row * cols];
        b[0] += y[row];

        for (size_t i = 0; i < cols; i++) {
            b[i + 1] += sample[i] * y[row];
        }

        a->data[0] += 1.0;
        for (size_t i = 0; i < cols; i++) {
            a->data[i + 1] += sample[i];
            a->data[(i + 1) * a_cols] += sample[i];

            for (size_t j = 0; j < cols; j++) {
                a->data[(i + 1) * a_cols + (j + 1)] += sample[i] * sample[j];
            }
        }
    }
}

Linear_Regression_Model train_model(const DataSet *dataset) {
    Linear_Regression_Model model;

    size_t cols = dataset->feature_matrix.cols;

    Matrix a = create_empty_matrix(cols + 1, cols + 1);
    double *b = calloc(cols + 1, sizeof(*b));

    build_normal_equation(dataset, &a, b);

    double *result_vector = malloc((cols + 1) * sizeof(*result_vector));
    if (!result_vector) {
        perror("Failed to allocate result_vector array");
        return model;
    }

    solve_linear_system(&a, b, result_vector);

    destroy_matrix(&a);
    free(b);

    model.parameters = result_vector;
    model.parameters_count = cols + 1;

    return model;
}

double predict(const double *x, const Linear_Regression_Model *model) {
    // Add intercept
    double result = model->parameters[0];

    for (size_t i = 1; i < model->parameters_count; i++) {
        result += x[i - 1] * model->parameters[i];
    }

    return result;
}

void print_model(Linear_Regression_Model *model) {
    printf("intercept = %.3f\n", model->parameters[0]);

    printf("Weights = [ ");
    for (size_t i = 1; i < model->parameters_count; i++) {
        printf("%.3f  ", model->parameters[i]);
    }

    printf("]\n");
}

void destroy_model(Linear_Regression_Model *model) {
    free(model->parameters);
    model->parameters = NULL;
    model->parameters_count = 0;
}

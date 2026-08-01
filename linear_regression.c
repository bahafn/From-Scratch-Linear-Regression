#include "linear_regression.h"

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

// TODO: This currently reads the file twice to get the number of rows and cols
//       and then adds the data. Check if there is a better way.
DataSet *read_csv_dataset(const char *file_path, size_t target_column_index) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file at %s\n", file_path);
        return NULL;
    }

    char line_buffer[MAX_LINE_LENGTH];
    size_t file_rows = 0;
    size_t file_cols = 0;

    // Count number of cols from the first line
    if (fgets(line_buffer, sizeof(line_buffer), file)) {
        file_rows++;

        char *ptr = line_buffer;
        while (*ptr) {
            if (*ptr == ',') {
                file_cols++;
            }

            ptr++;
        }
    }

    // Count number of rows in first pass
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        file_rows++;
    }

    rewind(file);
    DataSet *dataset = malloc(sizeof(DataSet));
    init_dataset(dataset, file_rows, file_cols);

    size_t row = 0;
    size_t dataset_index = 0;
    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        char *token = strtok(line_buffer, ",");

        for (size_t col = 0; col <= file_cols; col++) {
            if (col != target_column_index) {
                // NOTE: Currently we assume everything in the file is a number, will change later.
                dataset->feature_matrix.data[dataset_index++] = strtod(token, NULL);
            } else {
                dataset->target_vector[row] = strtod(token, NULL);
            }

            token = strtok(NULL, ",");
        }

        row++;
    }

    fclose(file);
    return dataset;
}

Min_Max_Scaler_Set min_max_fit(const DataSet *dataset,
                               size_t column_count,
                               const size_t columns[]) {

    const Matrix *feature_matrix = &dataset->feature_matrix;

    size_t rows = feature_matrix->rows;
    size_t cols = feature_matrix->cols;

    Min_Max_Scaler_Set scaler_set;
    scaler_set.scaler_count = column_count;
    scaler_set.scalers      = malloc(column_count * sizeof(*scaler_set.scalers));

    if (!scaler_set.scalers) {
        scaler_set.scaler_count = 0;
        return scaler_set;
    }

    for (size_t i = 0; i < column_count; i++) {
        size_t col = columns ? columns[i] : i;

        double col_max = -INFINITY;
        double col_min = INFINITY;

        for (size_t row = 0; row < rows; row++) {
            double value = feature_matrix->data[row * cols + col];

            if (value < col_min) {
                col_min = value;
            }
            if (value > col_max) {
                col_max = value;
            }
        }

        Min_Max_Scaler scaler = {
            .min_value    = col_min,
            .max_value    = col_max,
            .col_index = col
        };
        scaler_set.scalers[i] = scaler;
    }

    return scaler_set;
}

void min_max_transform(DataSet *dataset, Min_Max_Scaler_Set *scaler_set) {
    Matrix *matrix = &dataset->feature_matrix;

    size_t rows = matrix->rows;
    size_t cols = matrix->cols;

    for (size_t i = 0; i < scaler_set->scaler_count; i++) {
        Min_Max_Scaler *scaler = &scaler_set->scalers[i];

        size_t col   = scaler->col_index;
        double range = scaler->max_value - scaler->min_value;
        for (size_t row = 0; row < rows; row++) {
            size_t index = row * cols + col;

            if (scaler->min_value == scaler->max_value) {
                matrix->data[index] = 0;
            } else {
                matrix->data[index] = (matrix->data[index] - scaler->min_value) / range;
            }
        }
    }
}

void min_max_transform_row(double *row, Min_Max_Scaler_Set *scaler_set) {
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

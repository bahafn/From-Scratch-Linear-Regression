#include "linear_regression.h"
#include "matrix.h"
#include "string_utils.h"
#include "dynamic_array.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

Dataset create_empty_dataset(size_t samples, size_t features) {
    Dataset dataset;

    dataset.feature_matrix = create_empty_matrix(samples, features);
    dataset.target_vector  = malloc(sizeof(*dataset.target_vector) * samples);

    return dataset;
}

Dataset create_dataset(size_t samples, size_t features,
                       double (*feature_matrix)[features],
                       double *target_vector) {

    Dataset dataset;

    dataset.feature_matrix = create_matrix(samples, features, feature_matrix);
    dataset.target_vector  = target_vector;

    return dataset;
}

void init_dataset(Dataset *dataset, size_t samples, size_t features) {
    dataset->feature_matrix = create_empty_matrix(samples, features);
    dataset->target_vector  = malloc(samples * sizeof(*dataset->target_vector));
}

void destroy_dataset(Dataset *dataset) {
    if (dataset) {
        destroy_matrix(&dataset->feature_matrix);
        free(dataset->target_vector);
        dataset->target_vector = NULL;
    }
}

#define MAX_LINE_LENGTH 65536

Dataset *read_csv_dataset(const char *file_path, size_t target_column_index) {
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

        char *token = strtok_single(line_buffer, ",");
        while (token) {
            cols++;

            double value = string_to_double(token);
            if (isnan(value)) {
                fprintf(stderr, "Non double value in dataset: value = %s\n"
                                "Any results from this dataset should not be trusted.\n", token);
            }

            if (cols - 1 != target_column_index) {
                da_append(data, value);
            } else {
                first_in_target_vector = value;
            }

            token = strtok_single(NULL, ",");
        }
    } else {
        fprintf(stderr, "File is empty: %s\n", file_path);
        fclose(file);
        return NULL;
    }

    // Currently has the estimated number of rows depending on the estimated_entries and cols
    double_da target_vector;
    da_init(target_vector, estimated_entries / cols);
    da_append(target_vector, first_in_target_vector);

    while (fgets(line_buffer, sizeof(line_buffer), file)) {
        char *token = strtok_single(line_buffer, ",");

        for (size_t col = 0; col < cols; col++) {
            double value = string_to_double(token);
            if (isnan(value)) {
                fprintf(stderr, "Non double value in dataset: value = %s\n"
                                "Any results from this dataset should not be trusted.\n", token);
            }

            if (col != target_column_index) {
                da_append(data, value);
            } else {
                da_append(target_vector, value);
            }

            token = strtok_single(NULL, ",");
        }

        rows++;
    }

    da_resize(data, rows * (cols - 1));
    da_resize(target_vector, rows);

    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->target_vector       = target_vector.items;
    dataset->feature_matrix.rows = rows;
    dataset->feature_matrix.cols = cols - 1; // - 1 because we counted the target_vector
    dataset->feature_matrix.data = data.items;

    fclose(file);
    return dataset;
}

static void shuffle_indices(size_t n, size_t *indices) {
    for (size_t i = n; i > 1; i--) {
        size_t j = rand() % i;

        size_t temp = indices[i - 1];
        indices[i - 1] = indices[j];
        indices[j] = temp;
    }
}

Split_Dataset train_test_split(Dataset *dataset, double test_ratio, unsigned int random_state) {
    assert(test_ratio > 0 && test_ratio < 1 && "test_ratio must be between 0 and 1");

    srand(random_state);

    size_t test_rows  = round(dataset->feature_matrix.rows * test_ratio);
    size_t train_rows = dataset->feature_matrix.rows - test_rows;

    size_t samples  = dataset->feature_matrix.rows;
    size_t features = dataset->feature_matrix.cols;

    Dataset train = create_empty_dataset(train_rows, features);
    Dataset test  = create_empty_dataset(test_rows,  features);

    size_t *indices = malloc(sizeof(*indices) * samples);
    for (size_t i = 0; i < samples; i++) {
        indices[i] = i;
    }

    shuffle_indices(samples, indices);

    // Add first test_ratio percent rows to test dataset
    for (size_t i = 0; i < test_rows; i++) {
        size_t row_index = indices[i];

        memcpy(test.feature_matrix.data + i * features,
               dataset->feature_matrix.data + row_index * features,
               features * sizeof(double));
        test.target_vector[i] = dataset->target_vector[row_index];
    }

    // Add rest of rows to train dataset
    for (size_t i = test_rows; i < samples; i++) {
        size_t train_index = i - test_rows;
        size_t row_index = indices[i];

        memcpy(train.feature_matrix.data + train_index * features,
               dataset->feature_matrix.data + row_index * features,
               features * sizeof(double));
        train.target_vector[train_index] = dataset->target_vector[row_index];
    }

    free(indices);

    Split_Dataset split_dataset = {
        .train = train,
        .test  = test
    };
    return split_dataset;
}

void destroy_split_dataset(Split_Dataset *split_dataset) {
    if (split_dataset) {
        destroy_dataset(&split_dataset->train);
        destroy_dataset(&split_dataset->test);
    }
}

Min_Max_Scaler_Set min_max_fit(const Dataset *dataset,
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


Min_Max_Scaler_Set min_max_fit_all(const Dataset *dataset) {
    return min_max_fit(dataset, dataset->feature_matrix.cols, NULL);
}

void min_max_transform(Dataset *dataset, const Min_Max_Scaler_Set *scaler_set) {
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
    if (scaler_set) {
        free(scaler_set->scalers);
    }
}

static void build_normal_equation(const Dataset *dataset, Matrix *a, double *b) {
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

Linear_Regression_Model train_model(const Dataset *dataset) {
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

double *predict_all(const Matrix *feature_matrix, const Linear_Regression_Model *model) {
    double *result = malloc(feature_matrix->rows * sizeof(*result));

    for (size_t i = 0; i < feature_matrix->rows; i++) {
        result[i] = predict(feature_matrix->data + i * feature_matrix->cols, model);
    }

    return result;
}

static double calculate_error(const double *y_pred,
                              const double *y_true,
                              size_t n,
                              double (*transform)(double)) {

    if (n == 0) {
        return 0;
    }

    double error = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = y_true[i] - y_pred[i];
        error += transform(diff);
    }

    return error / n;
}

double calculate_mae(const double *y_pred, const double *y_true, size_t n) {
    return calculate_error(y_pred, y_true, n, fabs);
}

// Used as the function pointer in calculate_mse
static double squared_error(double error) {
    return error * error;
}

double calculate_mse(const double *y_pred, const double *y_true, size_t n) {
    return calculate_error(y_pred, y_true, n, squared_error);
}

double calculate_rmse(const double *y_pred, const double *y_true, size_t n) {
    return sqrt(calculate_mse(y_pred, y_true, n));
}

// NOTE: This implementation skips any difference value equaled to zero
double calculate_mape(const double *y_pred, const double *y_true, size_t n) {
    double error = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < n; i++) {
        if (y_true[i] == 0.0) {
            continue;
        }

        error += fabs((y_true[i] - y_pred[i]) / y_true[i]);
        count++;
    }

    return count == 0 ? 0.0 : (error / count) * 100.0;
}

double calculate_r2(const double *y_pred, const double *y_true, size_t n) {
    if (n == 0) {
        return 0.0;
    }

    double mean = 0.0;

    for (size_t i = 0; i < n; i++) {
        mean += y_true[i];
    }

    mean /= n;

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for (size_t i = 0; i < n; i++) {
        double residual  = y_true[i] - y_pred[i];
        double deviation = y_true[i] - mean;

        ss_res += residual * residual;
        ss_tot += deviation * deviation;
    }

    if (ss_tot == 0.0) {
        return 0.0;
    }

    return 1.0 - (ss_res / ss_tot);
}

Linear_Regression_Metrics calculate_linear_regression_metrics(
    const double *y_pred,
    const double *y_true,
    size_t n
) {

    Linear_Regression_Metrics metrics;

    metrics.mae  = calculate_mae(y_pred, y_true, n);
    metrics.mse  = calculate_mse(y_pred, y_true, n);
    metrics.rmse = sqrt(metrics.mse);
    metrics.r2   = calculate_r2(y_pred, y_true, n);
    metrics.mape = calculate_mape(y_pred, y_true, n);

    return metrics;
}

void print_linear_regression_metrics(const Linear_Regression_Metrics *metrics) {
    printf("MAE  = %f\n", metrics->mae);
    printf("MSE  = %f\n", metrics->mse);
    printf("RMSE = %f\n", metrics->rmse);
    printf("R2   = %f\n", metrics->r2);
    printf("MAPE = %f\n", metrics->mape);
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

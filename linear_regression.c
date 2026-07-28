#include "linear_regression.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

DataSet create_dataset(size_t samples, size_t features,
                       double (*feature_matrix)[features],
                       double *target_vector) {

    DataSet dataset;

    dataset.feature_matrix = create_matrix(samples, features, feature_matrix);
    dataset.target_vector  = target_vector;

    return dataset;
}

void destroy_dataset(DataSet *dataset) {
    destroy_matrix(&dataset->feature_matrix);
    dataset->target_vector = NULL;
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

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

// Adds a column at the start of a 2d array (destination[i][0] to all ones for use in
// calculating the bias during linear regression calculations.
// TODO: Clean up this function (doesn't make sense to take source as 2d array and destination as 1d, maybe make destination as a matrix)
static void add_bias_column(size_t rows, size_t cols,
                            const double *source,
                            double *destination) {

    size_t d_cols = cols + 1;

    for (size_t i = 0; i < rows; i++) {
        size_t index = i * d_cols;
        destination[index] = 1.0;
        memcpy(&destination[index + 1], &source[i * cols], sizeof(double) * cols);
    }
}

Linear_Regression_Model train_model(const DataSet *dataset) {
    size_t rows = dataset->feature_matrix.rows;
    size_t cols = dataset->feature_matrix.cols;

    Matrix xb = create_empty_matrix(rows, cols + 1); // Matrix from x with a bias column
    add_bias_column(rows, cols, dataset->feature_matrix.data, xb.data);

    Matrix x_transpose = matrix_transpose(&xb);

    Matrix  a = matrix_multiply(&x_transpose, &xb);
    double *b = matrix_vector_multiply(&x_transpose, dataset->target_vector);

    double *result_vector = malloc((cols + 1) * sizeof(*result_vector));
    solve_linear_system(&a, b, result_vector);

    destroy_matrix(&xb);
    destroy_matrix(&x_transpose);
    destroy_matrix(&a);
    free(b);

    Linear_Regression_Model model;
    model.parameters       = result_vector;
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

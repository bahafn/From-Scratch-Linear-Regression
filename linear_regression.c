#include "linear_regression.h"
#include "matrix.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Adds a column at the start of a 2d array (destination[i][0] to all ones for use in
// calculating the bias during linear regression calculations.
// TODO: Clean up this function (doesn't make sense to take source as 2d array and destination as 1d, maybe make destination as a matrix)
static void add_bias_column(size_t rows, size_t cols,
                            const double source[rows][cols],
                            double *destination) {

    size_t d_cols = cols + 1;

    for (size_t i = 0; i < rows; i++) {
        size_t index = i * d_cols;
        destination[index] = 1.0;
        memcpy(&destination[index + 1], &source[i], sizeof(double) * cols);
    }
}

Linear_Regression_Model train_model(size_t rows, size_t cols,
                                    const double (*x)[cols],
                                    const double *y) {

    Matrix xb = create_matrix(rows, cols + 1); // Matrix from x with a bias column
    add_bias_column(rows, cols, x, xb.data);

    Matrix x_transpose = matrix_transpose(&xb);

    Matrix  a = matrix_multiply(&x_transpose, &xb);
    double *b = matrix_vector_multiply(&x_transpose, y);

    double *result_vector = malloc((cols + 1) * sizeof(*result_vector));
    solve_linear_system(&a, b, result_vector);

    destroy_matrix(&xb);
    destroy_matrix(&x_transpose);

    Linear_Regression_Model model;
    model.intercept     = result_vector[0];
    model.weights       = result_vector + 1;
    model.weights_count = cols;

    return model;
}

double predict(size_t cols, const double x[cols], const Linear_Regression_Model *model) {
    double result = model->intercept;

    for (size_t i = 0; i < cols; i++) {
        result += x[i] * model->weights[i];
    }

    return result;
}

void print_model(Linear_Regression_Model *model) {
    printf("intercept = %.3f\n", model->intercept);

    printf("Weights = [ ");
    for (size_t i = 0; i < model->weights_count; i++) {
        printf("%.3f  ", model->weights[i]);
    }

    printf("]\n");
}

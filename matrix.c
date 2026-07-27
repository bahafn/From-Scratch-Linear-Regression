#include "matrix.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// TODO: Fix using asserts for error handling.

Matrix create_empty_matrix(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;

    m.data = malloc(sizeof(*m.data) * rows * cols);

    return m;
}

Matrix create_matrix(size_t rows, size_t cols, const double (*data)[cols]) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;

    m.data = malloc(sizeof(*m.data) * rows * cols);

    if (data == NULL) {
        m.rows = 0;
        m.cols = 0;
        return m;
    }

    memcpy(m.data, data, rows * cols * sizeof(double));
    return m;
}

void destroy_matrix(Matrix *m) {
    m->rows = 0;
    m->cols = 0;
    free(m->data);
    m->data = NULL;
}

void print_matrix(const Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        printf("%.4f ", m->data[i]);

        if ((i + 1) % m->cols == 0) {
            printf("\n");
        }
    }

    printf("\n");
}

Matrix matrix_transpose(const Matrix *m) {
    Matrix result = create_empty_matrix(m->cols, m->rows);

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            *matrix_at(&result, j, i) = *matrix_at(m, i, j);
        }
    }

    return result;
}

Matrix matrix_multiply(const Matrix *m1, const Matrix *m2) {
    assert(m1->cols == m2->rows && "m1 and m2 can't be multiplied");

    Matrix result = create_empty_matrix(m1->rows, m2->cols);

    for (size_t i = 0; i < m1->rows; i++) {
        for (size_t j = 0; j < m2->cols; j++) {
            *matrix_at(&result, i, j) = 0.0;

            for (size_t k = 0; k < m1->cols; k++) {
                *matrix_at(&result, i, j) +=
                    *matrix_at_const(m1, i, k) *
                    *matrix_at_const(m2, k, j);
            }
        }
    }

    return result;
}

double *matrix_vector_multiply(const Matrix *m, const double *v) {
    double *result = calloc(m->rows, sizeof(*result));

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            result[i] += *matrix_at_const(m, i, j) * v[j];
        }
    }

    return result;
}

// TODO: Make sure it is the best way to do this in a for loop
static void matrix_swap_rows(Matrix *m, size_t row1, size_t row2) {
    if (row1 == row2) {
        return;
    }

    for (size_t col = 0; col < m->cols; col++) {
        double temp = *matrix_at(m, row1, col);
        *matrix_at(m, row1, col) = *matrix_at(m, row2, col);
        *matrix_at(m, row2, col) = temp;
    }
}

bool solve_linear_system(const Matrix *A, const double *b, double *x) {
    if (A->rows != A->cols) {
        return false;
    }

    size_t n = A->rows;
    Matrix aug = create_empty_matrix(n, n + 1);

    // Create augmented matrix [A | b]
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            *matrix_at(&aug, i, j) = *matrix_at_const(A, i, j);
        }

        *matrix_at(&aug, i, n) = b[i];
    }


    // Forward elimination
    for (size_t col = 0; col < n; col++) {
        size_t pivot = col;

        for (size_t row = col + 1; row < n; row++) {
            if (fabs(*matrix_at(&aug, row, col)) > fabs(*matrix_at(&aug, pivot, col))) {
                pivot = row;
            }
        }

        if (fabs(*matrix_at(&aug, pivot, col)) < 1e-12) {
            destroy_matrix(&aug);
            return false;
        }

        matrix_swap_rows(&aug, col, pivot);

        for (size_t row = col + 1; row < n; row++) {
            double factor = *matrix_at(&aug, row, col) / *matrix_at(&aug, col, col);
            for (size_t j = col; j <= n; j++) {
                *matrix_at(&aug, row, j) -= factor * (*matrix_at(&aug, col, j));
            }
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        double sum = *matrix_at(&aug, i, n);
        for (size_t j = i + 1; j < n; j++) {
            sum -= *matrix_at(&aug, i, j) * x[j];
        }

        x[i] = sum / *matrix_at(&aug, i, i);
    }

    destroy_matrix(&aug);
    return true;
}

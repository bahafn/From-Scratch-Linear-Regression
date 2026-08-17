#include <CUnit/Basic.h>
#include <stdlib.h>
#include "../src/matrix.h"

void test_empty_matrix_creation() {
    Matrix matrix = create_empty_matrix(3, 4);

    CU_ASSERT_EQUAL(matrix.rows, 3);
    CU_ASSERT_EQUAL(matrix.cols, 4);
    CU_ASSERT_PTR_NOT_NULL(matrix.data);

    destroy_matrix(&matrix);
}

void test_matrix_creation() {
    const double data[2][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix matrix = create_matrix(2, 3, data);

    CU_ASSERT_EQUAL(matrix.rows, 2);
    CU_ASSERT_EQUAL(matrix.cols, 3);
    CU_ASSERT_PTR_NOT_NULL(matrix.data);

    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&matrix, 0, 0), 1.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&matrix, 1, 2), 6.0, 1e-12);

    destroy_matrix(&matrix);
}

void test_matrix_multiplication() {
    const double data1[2][2] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const double data2[2][2] = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    Matrix a = create_matrix(2, 2, data1);
    Matrix b = create_matrix(2, 2, data2);

    *matrix_at(&a, 0, 0) = 1;
    *matrix_at(&a, 0, 1) = 2;
    *matrix_at(&a, 1, 0) = 3;
    *matrix_at(&a, 1, 1) = 4;

    *matrix_at(&b, 0, 0) = 5;
    *matrix_at(&b, 0, 1) = 6;
    *matrix_at(&b, 1, 0) = 7;
    *matrix_at(&b, 1, 1) = 8;

    Matrix result = matrix_multiply(&a, &b);

    CU_ASSERT_EQUAL(result.rows, 2);
    CU_ASSERT_EQUAL(result.cols, 2);

    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&result, 0, 0), 19.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&result, 0, 1), 22.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&result, 1, 0), 43.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&result, 1, 1), 50.0, 1e-10);

    destroy_matrix(&a);
    destroy_matrix(&b);
    destroy_matrix(&result);
}

void test_matrix_vector_multiplication() {
    const double matrix_data[2][2] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    double vector[2] = { 5.0, 6.0 };

    Matrix  matrix = create_matrix(2, 2, matrix_data);
    double *result = matrix_vector_multiply(&matrix, vector);

    // CU_ASSERT_EQUAL(sizeof(result) / sizeof(*result), 2);

    CU_ASSERT_DOUBLE_EQUAL(result[0], 17.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(result[1], 39.0, 1e-12);

    destroy_matrix(&matrix);
    free(result);
}

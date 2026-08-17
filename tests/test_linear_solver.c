#include <CUnit/Basic.h>
#include "../src/matrix.h"

// 2x + y = 5
// x + 3y = 6
// Solution:
// x = 1.8
// y = 1.4
void test_solver_basic() {
    const double data[2][2] = {
        {2.0, 1.0},
        {1.0, 3.0}
    };

    double b[2] = { 5.0, 6.0 };

    double x[2];

    Matrix matrix = create_matrix(2, 2, data);

    bool result = solve_linear_system(&matrix, b, x);

    CU_ASSERT_TRUE(result);

    CU_ASSERT_DOUBLE_EQUAL(x[0], 1.8, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(x[1], 1.4, 1e-10);

    destroy_matrix(&matrix);
}

// 0x + 2y = 4
// 3x + 4y = 10
// Solution:
// x = 2
// y = 2
// The first pivot is zero, so the solver must swap the rows.
void test_solver_partial_pivoting() {
    const double data[2][2] = {
        {0.0, 2.0},
        {3.0, 4.0}
    };

    double b[2] = { 4.0, 10.0 };

    double x[2];

    Matrix matrix = create_matrix(2, 2, data);

    bool result = solve_linear_system(&matrix, b, x);

    CU_ASSERT_TRUE(result);

    CU_ASSERT_DOUBLE_EQUAL(x[0], 2.0 / 3.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(x[1], 2.0, 1e-10);

    destroy_matrix(&matrix);
}

// x + 2y = 3
// 2x + 4y = 6
// The second equation is just 2 times the first, so the matrix is singular.
void test_solver_singular_matrix() {
    const double data[2][2] = {
        {1.0, 2.0},
        {2.0, 4.0}
    };

    double b[2] = { 3.0, 6.0 };

    double x[2];

    Matrix matrix = create_matrix(2, 2, data);

    bool result = solve_linear_system(&matrix, b, x);

    CU_ASSERT_FALSE(result);

    destroy_matrix(&matrix);
}

// 2x +  y -  z =   8
// -3x -  y + 2z = -11
// -2x +  y + 2z =  -3
// Expected solution:
// x =  2
// y =  3
// z = -1
void test_solver_3x3() {
    const double data[3][3] = {
        {2.0, 1.0, -1.0},
        {-3.0, -1.0, 2.0},
        {-2.0, 1.0, 2.0}
    };

    double b[3] = {8.0, -11.0, -3.0};
    double x[3];

    Matrix matrix = create_matrix(3, 3, data);

    bool result = solve_linear_system(&matrix, b, x);

    CU_ASSERT_TRUE(result);

    CU_ASSERT_DOUBLE_EQUAL(x[0], 2.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(x[1], 3.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(x[2], -1.0, 1e-10);

    destroy_matrix(&matrix);
}

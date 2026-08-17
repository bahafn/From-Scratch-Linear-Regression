#include <stdio.h>
#include <CUnit/Basic.h>

void test_empty_matrix_creation(void);
void test_matrix_creation(void);
void test_matrix_multiplication(void);
void test_matrix_vector_multiplication(void);

void test_solver_basic(void);
void test_solver_partial_pivoting(void);
void test_solver_singular_matrix(void);
void test_solver_3x3(void);

void test_min_max_transform(void);
void test_min_max_transform_negative_values(void);
void test_min_max_transform_constant_column(void);
void test_min_max_transform_multiple_columns(void);
void test_min_max_transform_outside_training_range(void);

void test_linear_regression_simple(void);
void test_linear_regression_prediction(void);
void test_linear_regression_multiple_features(void);
void test_linear_regression_predict_all(void);

void test_calculate_mae(void);
void test_calculate_mse(void);
void test_calculate_rmse(void);
void test_calculate_metrics_perfect_predictions(void);
void test_calculate_r2(void);
void test_calculate_linear_regression_metrics(void);

void test_string_to_double_integer(void);
void test_string_to_double_decimal(void);
void test_string_to_double_negative(void);
void test_string_to_double_scientific(void);
void test_string_to_double_zero(void);


int main(void) {
#ifdef _WIN32
    freopen("NUL", "w", stderr);
#else
    freopen("/dev/null", "w", stderr);
#endif

    if (CU_initialize_registry() != CUE_SUCCESS) {
        return CU_get_error();
    }

    CU_pSuite matrix_suite     = CU_add_suite("Matrix Tests", NULL, NULL);
    CU_pSuite solver_suite     = CU_add_suite("Linear Solver Tests", NULL, NULL);
    CU_pSuite scaler_suite     = CU_add_suite("Scaler Tests", NULL, NULL);
    CU_pSuite regression_suite = CU_add_suite("Regression Tests", NULL, NULL);
    CU_pSuite metrics_suite    = CU_add_suite("Metrics Tests", NULL, NULL);
    CU_pSuite parser_suite     = CU_add_suite("Parser Tests", NULL, NULL);

    if (matrix_suite == NULL ||
        solver_suite == NULL ||
        scaler_suite == NULL ||
        regression_suite == NULL ||
        metrics_suite == NULL ||
        parser_suite == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (CU_add_test(matrix_suite,
                    "Create empty matrix",
                    test_empty_matrix_creation) == NULL ||
        CU_add_test(matrix_suite,
                    "Create matrix from data",
                    test_matrix_creation) == NULL ||
        CU_add_test(matrix_suite,
                    "Multiply two matrices",
                    test_matrix_multiplication) == NULL ||
        CU_add_test(matrix_suite,
                    "Multiply matrix by vector",
                    test_matrix_vector_multiplication) == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (CU_add_test(solver_suite,
                    "Solve basic linear system",
                    test_solver_basic) == NULL ||
        CU_add_test(solver_suite,
                    "Handle partial pivoting",
                    test_solver_partial_pivoting) == NULL ||
        CU_add_test(solver_suite,
                    "Reject singular matrix",
                    test_solver_singular_matrix) == NULL ||
        CU_add_test(solver_suite,
                    "Solve 3x3 linear system",
                    test_solver_3x3) == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (CU_add_test(scaler_suite,
                    "Min-max transform",
                    test_min_max_transform) == NULL ||
        CU_add_test(scaler_suite,
                    "Min-max transform with negative values",
                    test_min_max_transform_negative_values) == NULL ||
        CU_add_test(scaler_suite,
                    "Min-max transform constant column",
                    test_min_max_transform_constant_column) == NULL ||
        CU_add_test(scaler_suite,
                    "Min-max transform multiple columns",
                    test_min_max_transform_multiple_columns) == NULL ||
        CU_add_test(scaler_suite,
                    "Transform values outside training range",
                    test_min_max_transform_outside_training_range) == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (CU_add_test(regression_suite,
                    "Train simple linear regression",
                    test_linear_regression_simple) == NULL ||
        CU_add_test(regression_suite,
                    "Predict single sample",
                    test_linear_regression_prediction) == NULL ||
        CU_add_test(regression_suite,
                    "Train multiple-feature regression",
                    test_linear_regression_multiple_features) == NULL ||
        CU_add_test(regression_suite,
                    "Predict multiple samples",
                    test_linear_regression_predict_all) == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (CU_add_test(metrics_suite,
                    "Calculate MAE",
                    test_calculate_mae) == NULL ||
        CU_add_test(metrics_suite,
                    "Calculate MSE",
                    test_calculate_mse) == NULL ||
        CU_add_test(metrics_suite,
                    "Calculate RMSE",
                    test_calculate_rmse) == NULL ||
        CU_add_test(metrics_suite,
                    "Perfect predictions",
                    test_calculate_metrics_perfect_predictions) == NULL ||
        CU_add_test(metrics_suite,
                    "Calculate R-squared",
                    test_calculate_r2) == NULL ||
        CU_add_test(metrics_suite,
                    "Calculate all regression metrics",
                    test_calculate_linear_regression_metrics) == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (CU_add_test(parser_suite,
                    "Parse integer",
                    test_string_to_double_integer) == NULL ||
        CU_add_test(parser_suite,
                    "Parse decimal",
                    test_string_to_double_decimal) == NULL ||
        CU_add_test(parser_suite,
                    "Parse negative number",
                    test_string_to_double_negative) == NULL ||
        CU_add_test(parser_suite,
                    "Parse scientific notation",
                    test_string_to_double_scientific) == NULL ||
        CU_add_test(parser_suite,
                    "Parse zero",
                    test_string_to_double_zero) == NULL)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();

    unsigned int failures = CU_get_number_of_failures();

    CU_cleanup_registry();

    return failures == 0 ? 0 : 1;
}

#include <CUnit/Basic.h>

#include "../src/linear_regression.h"

void test_calculate_mae() {
    const double y_true[3] = { 1.0, 2.0, 3.0 };
    const double y_pred[3] = { 2.0, 4.0, 2.0 };

    double result = calculate_mae(y_pred, y_true, 3);

    CU_ASSERT_DOUBLE_EQUAL(result, 4.0 / 3.0, 1e-12);
}

void test_calculate_mse() {
    const double y_true[3] = { 1.0, 2.0, 3.0 };
    const double y_pred[3] = { 2.0, 4.0, 2.0 };

    double result = calculate_mse(y_pred, y_true, 3);

    CU_ASSERT_DOUBLE_EQUAL(result, 2.0, 1e-12);
}

void test_calculate_rmse() {
    const double y_true[3] = { 1.0, 2.0, 3.0 };
    const double y_pred[3] = { 2.0, 4.0, 2.0 };

    double result = calculate_rmse(y_pred, y_true, 3);

    CU_ASSERT_DOUBLE_EQUAL(result, sqrt(2.0), 1e-12);
}

void test_calculate_metrics_perfect_predictions() {
    const double y[4] = { 1.0, 2.0, 3.0, 4.0 };

    CU_ASSERT_DOUBLE_EQUAL(calculate_mae(y, y, 4), 0.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(calculate_mse(y, y, 4), 0.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(calculate_rmse(y, y, 4), 0.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(calculate_r2(y, y, 4), 1.0, 1e-12);
}

void test_calculate_r2() {
    const double y_true[4] = { 1.0, 2.0, 3.0, 4.0 };
    const double y_pred[4] = { 1.0, 2.0, 3.0, 3.0 };

    double result = calculate_r2(y_pred, y_true, 4);

    CU_ASSERT_DOUBLE_EQUAL(result, 0.8, 1e-12);
}

void test_calculate_linear_regression_metrics() {
    const double y_true[4] = { 1.0, 2.0, 3.0, 4.0 };
    const double y_pred[4] = { 1.0, 2.0, 3.0, 3.0 };

    Linear_Regression_Metrics metrics = calculate_linear_regression_metrics(y_pred, y_true, 4);

    CU_ASSERT_DOUBLE_EQUAL(metrics.mae, 0.25, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(metrics.mse, 0.25, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(metrics.rmse, 0.5, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(metrics.r2, 0.8, 1e-12);
}

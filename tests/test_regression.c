#include <CUnit/Basic.h>
#include <stdlib.h>

#include "../src/linear_regression.h"

// Solves:
//   y = 3 + 2x
//
// Dataset:
//   x    y
//   0    3
//   1    5
//   2    7
//   3    9
//   4   11
void test_linear_regression_simple() {
    const double features[5][1] = {
        {0.0},
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    };

    double target_vector[5] = { 3.0, 5.0, 7.0, 9.0, 11.0 };

    Dataset dataset = create_dataset(5, 1, features, target_vector);

    Linear_Regression_Model model;

    bool result = train_model(&dataset, &model);

    CU_ASSERT_TRUE(result);
    CU_ASSERT_EQUAL(model.parameters_count, 2);
    CU_ASSERT_PTR_NOT_NULL(model.parameters);

    CU_ASSERT_DOUBLE_EQUAL(model.parameters[0], 3.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(model.parameters[1], 2.0, 1e-10);

    destroy_model(&model);
}

// Model:
//   y = 3 + 2x
//
// Predict:
//   x = 5 -> y = 13
void test_linear_regression_prediction() {
    const double features[5][1] = {
        {0.0},
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    };

    double target_vector[5] = { 3.0, 5.0, 7.0, 9.0, 11.0 };

    Dataset dataset = create_dataset(5, 1, features, target_vector);

    Linear_Regression_Model model;

    CU_ASSERT_TRUE(train_model(&dataset, &model));

    const double x[1] = {5.0};

    double prediction = predict(x, &model);

    CU_ASSERT_DOUBLE_EQUAL(prediction, 13.0, 1e-10);

    destroy_model(&model);
}

// Solves:
//   y = 5 + 2x1 + 3x2
void test_linear_regression_multiple_features() {
    const double features[5][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {2.0, 3.0}
    };

    double target_vector[5] = { 5.0, 7.0, 8.0, 10.0, 18.0 };

    Dataset dataset = create_dataset(5, 2, features, target_vector);

    Linear_Regression_Model model;

    bool result = train_model(&dataset, &model);

    CU_ASSERT_TRUE(result);
    CU_ASSERT_EQUAL(model.parameters_count, 3);

    CU_ASSERT_DOUBLE_EQUAL(model.parameters[0], 5.0, 1e-8);
    CU_ASSERT_DOUBLE_EQUAL(model.parameters[1], 2.0, 1e-8);
    CU_ASSERT_DOUBLE_EQUAL(model.parameters[2], 3.0, 1e-8);

    destroy_model(&model);
}

void test_linear_regression_predict_all() {
    const double features[4][1] = {
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    };

    double target_vector[4] = { 3.0, 5.0, 7.0, 9.0 };

    Dataset dataset = create_dataset(4, 1, features, target_vector);

    Linear_Regression_Model model = {0};

    CU_ASSERT_TRUE(train_model(&dataset, &model));

    double *predictions = predict_all(&dataset.feature_matrix, &model);

    CU_ASSERT_PTR_NOT_NULL(predictions);

    CU_ASSERT_DOUBLE_EQUAL(predictions[0], 3.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(predictions[1], 5.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(predictions[2], 7.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(predictions[3], 9.0, 1e-10);

    free(predictions);

    destroy_model(&model);
}

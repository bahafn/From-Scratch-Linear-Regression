#include <CUnit/Basic.h>
#include "../src/linear_regression.h"

void test_min_max_transform() {
    const double features[4][1] = {
        {1.0},
        {2.0},
        {3.0},
        {5.0}
    };

    Dataset dataset = create_dataset(4, 1, features, NULL);

    Min_Max_Scaler_Set scaler_set = min_max_fit_all(&dataset);

    CU_ASSERT_EQUAL(scaler_set.scaler_count, 1);
    CU_ASSERT_PTR_NOT_NULL(scaler_set.scalers);

    CU_ASSERT_DOUBLE_EQUAL(scaler_set.scalers[0].min_value, 1.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(scaler_set.scalers[0].max_value, 5.0, 1e-12);

    min_max_transform(&dataset, &scaler_set);

    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 0, 0), 0.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 1, 0), 0.25, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 2, 0), 0.5, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 3, 0), 1.0, 1e-12);

    destroy_scaler_set(&scaler_set);
    destroy_dataset(&dataset);
}

void test_min_max_transform_negative_values() {
    const double features[5][1] = {
        {-10.0},
        {-5.0},
        {0.0},
        {5.0},
        {10.0}
    };

    Dataset dataset = create_dataset(5, 1, features, NULL);

    Min_Max_Scaler_Set scaler_set = min_max_fit_all(&dataset);
    min_max_transform(&dataset, &scaler_set);

    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 0, 0), 0.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 1, 0), 0.25, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 2, 0), 0.5, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 3, 0), 0.75, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, 4, 0), 1.0, 1e-12);

    destroy_scaler_set(&scaler_set);
    destroy_dataset(&dataset);
}

void test_min_max_transform_constant_column() {
    const double features[4][1] = {
        {5.0},
        {5.0},
        {5.0},
        {5.0}
    };

    Dataset dataset = create_dataset(4, 1, features, NULL);

    Min_Max_Scaler_Set scaler_set = min_max_fit_all(&dataset);

    min_max_transform(&dataset, &scaler_set);

    for (size_t i = 0; i < 4; i++) {
        CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, i, 0), 0.0, 1e-12);
    }

    destroy_scaler_set(&scaler_set);
    destroy_dataset(&dataset);
}

void test_min_max_transform_multiple_columns() {
    const double features[3][2] = {
        {0.0, 10.0},
        {5.0, 20.0},
        {10.0, 30.0}
    };

    Dataset dataset = create_dataset(3, 2, features, NULL);

    Min_Max_Scaler_Set scaler_set = min_max_fit_all(&dataset);

    CU_ASSERT_EQUAL(scaler_set.scaler_count, 2);

    min_max_transform(&dataset, &scaler_set);

    for (size_t i = 0; i < 3; i++) {
        double expected = i * 0.5;
        CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, i, 0), expected, 1e-12);
        CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&dataset.feature_matrix, i, 1), expected, 1e-12);
    }

    destroy_scaler_set(&scaler_set);
    destroy_dataset(&dataset);
}

void test_min_max_transform_outside_training_range() {
    const double training_features[2][1] = {
        {0.0},
        {10.0}
    };

    const double test_features[2][1] = {
        {-10.0},
        {20.0}
    };

    Dataset training = create_dataset(2, 1, training_features, NULL);
    Dataset test     = create_dataset(2, 1, test_features, NULL);

    Min_Max_Scaler_Set scaler_set = min_max_fit_all(&training);

    min_max_transform(&test, &scaler_set);

    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&test.feature_matrix, 0, 0), -1.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(*matrix_at_const(&test.feature_matrix, 1, 0), 2.0, 1e-12);

    destroy_scaler_set(&scaler_set);
    destroy_dataset(&training);
    destroy_dataset(&test);
}

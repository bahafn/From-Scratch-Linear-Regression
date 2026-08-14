#include <stdlib.h>

#include "linear_regression.h"

int main() {
    DataSet *dataset = read_csv_dataset("test_datasets/test_dataset_2.csv", 0);
    if (!dataset) {
        return 0;
    }

    Split_DataSet split_dataset = train_test_split(dataset, 0.2, 42);

    // size_t cols[] = { 0, 4, 6, 10, 12, 51, 13, 20, 21, 13 };
    // Min_Max_Scaler_Set scaler_set = min_max_fit(dataset, 3, cols);
    Min_Max_Scaler_Set scaler_set = min_max_fit_all(&split_dataset.train);
    min_max_transform(&split_dataset.train, &scaler_set);
    min_max_transform(&split_dataset.test,  &scaler_set);

    Linear_Regression_Model model = train_model(&split_dataset.train);
    print_model(&model);

    double *y_pred = predict_matrix(&split_dataset.test.feature_matrix, &model);
    Linear_Regression_Metrics metrics = calculate_linear_regression_metrics(
        y_pred,
        split_dataset.test.target_vector,
        split_dataset.test.feature_matrix.rows
    );

    print_linear_regression_metrics(&metrics);

    destroy_dataset(dataset);
    free(dataset);
    destroy_model(&model);
    destroy_scaler_set(&scaler_set);

    return 0;
}

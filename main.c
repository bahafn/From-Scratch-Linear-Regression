#include <stdio.h>
#include <stdlib.h>

#include "linear_regression.h"

int main() {
    DataSet *dataset = read_csv_dataset("test_datasets/winequality-white.csv.txt", 0);
    if (!dataset) {
        return 0;
    }

    // size_t cols[] = { 0, 4, 6, 10, 12, 51, 13, 20, 21, 13 };
    // Min_Max_Scaler_Set scaler_set = min_max_fit(dataset, 3, cols);
    Min_Max_Scaler_Set scaler_set = min_max_fit_all(dataset);
    min_max_transform(dataset, &scaler_set);

    Linear_Regression_Model model = train_model(dataset);
    print_model(&model);

    double x_predict[] = { 6.5,0.24,0.19,1.2,0.041,30,111,0.99254,2.99,0.46,9.4,6 };
    min_max_transform_row(x_predict, &scaler_set);

    printf("Price estimated: %f\n", predict(x_predict, &model));

    destroy_dataset(dataset);
    free(dataset);
    destroy_model(&model);
    destroy_scaler_set(&scaler_set);

    return 0;
}

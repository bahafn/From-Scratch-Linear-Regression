#include "matrix.h"

#include <stddef.h>

typedef struct {
    double *target_vector;
    Matrix  feature_matrix;
} DataSet;

DataSet create_dataset(size_t samples, size_t features,
                       double (*feature_matrix)[features],
                       double *target_vector);
void     init_dataset(DataSet *dataset, size_t samples, size_t features);
void     destroy_dataset(DataSet *dataset);
DataSet *read_csv_dataset(const char *file_path, size_t target_column_index);

typedef struct {
    double min_value;
    double max_value;
    size_t col_index;
} Min_Max_Scaler;

typedef struct {
    size_t scaler_count;
    Min_Max_Scaler *scalers;
} Min_Max_Scaler_Set;

Min_Max_Scaler_Set min_max_fit(const DataSet *dataset,
                               size_t column_count,
                               const size_t columns[]);

static inline Min_Max_Scaler_Set min_max_fit_all(const DataSet *dataset) {
    return min_max_fit(dataset, dataset->feature_matrix.cols, NULL);
}

void min_max_transform(DataSet *dataset, Min_Max_Scaler_Set *scaler_set);
void min_max_transform_row(double *row, Min_Max_Scaler_Set *scaler_set);

void destroy_scaler_set(Min_Max_Scaler_Set *scaler_set);

typedef struct {
    // First element of this is the intercept and the other elements are the weights of each feature
    double *parameters;
    size_t parameters_count;
} Linear_Regression_Model;

Linear_Regression_Model train_model(const DataSet *dataset);
double predict(const double *x, const Linear_Regression_Model *model);

void print_model(Linear_Regression_Model *model);
void destroy_model(Linear_Regression_Model *model);

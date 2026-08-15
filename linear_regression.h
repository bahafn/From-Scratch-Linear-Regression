#pragma once

#include "matrix.h"

typedef struct {
    double *target_vector;
    Matrix  feature_matrix;
} Dataset;

Dataset create_empty_dataset(size_t samples, size_t features);
Dataset create_dataset(size_t samples, size_t features,
                       double (*feature_matrix)[features],
                       double *target_vector);
void     init_dataset(Dataset *dataset, size_t samples, size_t features);
void     destroy_dataset(Dataset *dataset);
Dataset *read_csv_dataset(const char *file_path, size_t target_column_index);

typedef struct {
    Dataset train;
    Dataset test;
} Split_Dataset;

Split_Dataset train_test_split(Dataset *dataset, double test_ratio, int random_state);
void destroy_split_dataset(Split_Dataset *split_dataset);

typedef struct {
    double min_value;
    double max_value;
    size_t col_index;
} Min_Max_Scaler;

typedef struct {
    size_t scaler_count;
    Min_Max_Scaler *scalers;
} Min_Max_Scaler_Set;

Min_Max_Scaler_Set min_max_fit(const Dataset *dataset,
                               size_t column_count,
                               const size_t columns[]);
Min_Max_Scaler_Set min_max_fit_all(const Dataset *dataset);

void min_max_transform(Dataset *dataset, const Min_Max_Scaler_Set *scaler_set);
void min_max_transform_row(double *row, const Min_Max_Scaler_Set *scaler_set);

void destroy_scaler_set(Min_Max_Scaler_Set *scaler_set);

typedef struct {
    // First element of this is the intercept and the other elements are the weights of each feature
    double *parameters;
    size_t parameters_count;
} Linear_Regression_Model;

Linear_Regression_Model train_model(const Dataset *dataset);
double  predict(const double *x, const Linear_Regression_Model *model);
double *predict_all(const Matrix *feature_matrix, const Linear_Regression_Model *model);

void print_model(Linear_Regression_Model *model);
void destroy_model(Linear_Regression_Model *model);

typedef struct {
    double mae;
    double mse;
    double rmse;
    double r2;
    double mape;
} Linear_Regression_Metrics;

double calculate_mae(const double *y_pred, const double *y_true, size_t n);
double calculate_mse(const double *y_pred, const double *y_true, size_t n);
double calculate_rmse(const double *y_pred, const double *y_true, size_t n);
double calculate_mape(const double *y_pred, const double *y_true, size_t n);
double calculate_r2(const double *y_pred, const double *y_true, size_t n);

Linear_Regression_Metrics calculate_linear_regression_metrics(
        const double *y_pred,
        const double *y_true,
        size_t n
);

void print_linear_regression_metrics(const Linear_Regression_Metrics *metrics);

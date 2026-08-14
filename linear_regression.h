#pragma once

#include "matrix.h"

typedef struct {
    double *target_vector;
    Matrix  feature_matrix;
} DataSet;

DataSet create_empty_dataset(size_t samples, size_t features);
DataSet create_dataset(size_t samples, size_t features,
                       double (*feature_matrix)[features],
                       double *target_vector);
void     init_dataset(DataSet *dataset, size_t samples, size_t features);
void     destroy_dataset(DataSet *dataset);
DataSet *read_csv_dataset(const char *file_path, size_t target_column_index);

typedef struct {
    DataSet train;
    DataSet test;
} Split_DataSet;

Split_DataSet train_test_split(DataSet *dataset, double test_ratio, int random_state);

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
Min_Max_Scaler_Set min_max_fit_all(const DataSet *dataset);

void min_max_transform(DataSet *dataset, const Min_Max_Scaler_Set *scaler_set);
void min_max_transform_row(double *row, const Min_Max_Scaler_Set *scaler_set);

void destroy_scaler_set(Min_Max_Scaler_Set *scaler_set);

typedef struct {
    // First element of this is the intercept and the other elements are the weights of each feature
    double *parameters;
    size_t parameters_count;
} Linear_Regression_Model;

Linear_Regression_Model train_model(const DataSet *dataset);
double predict(const double *x, const Linear_Regression_Model *model);
double *predict_matrix(const Matrix *feature_matrix, const Linear_Regression_Model *model);

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


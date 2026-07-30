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
    // First element of this is the intercept and the other elements are the weights of each feature
    double *parameters;
    size_t parameters_count;
} Linear_Regression_Model;

Linear_Regression_Model train_model(const DataSet *dataset);
double predict(const double *x, const Linear_Regression_Model *model);

void print_model(Linear_Regression_Model *model);
void destroy_model(Linear_Regression_Model *model);

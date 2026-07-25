#include <stddef.h>

typedef struct {
    double  intercept;
    double *weights;

    size_t weights_count;
} Linear_Regression_Model;

Linear_Regression_Model train_model(size_t rows, size_t cols,
                                    const double (*x)[cols],
                                    const double *y);
double predict(size_t cols, const double x[cols], const Linear_Regression_Model *model);

void print_model(Linear_Regression_Model *model);

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    size_t rows;
    size_t cols;

    double *data;
} Matrix;

Matrix create_empty_matrix(size_t rows, size_t cols);
Matrix create_matrix(size_t rows, size_t cols, const double (*data)[cols]);
void   destroy_matrix(Matrix *m);
void   print_matrix(const Matrix *m);

Matrix  matrix_transpose(const Matrix *m);
Matrix  matrix_multiply(const Matrix *m1, const Matrix *m2);
double *matrix_vector_multiply(const Matrix *m, const double *v);

bool solve_linear_system(const Matrix *A, const double *b, double *x);

inline double *matrix_at(const Matrix *m, size_t i, size_t j) {
    return &m->data[i * m->cols + j];
}

inline const double *matrix_at_const(const Matrix *m, size_t i, size_t j) {
    return &m->data[i * m->cols + j];
}

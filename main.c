#include <stdio.h>

#include "linear_regression.h"
#include "matrix.h"

int main() {
    double x[8][3] = {
        {2100, 3, 20},
        {1600, 2, 15},
        {2400, 4, 25},
        {1400, 2, 10},
        {3000, 5, 30},
        {1800, 3, 18},
        {2600, 4, 22},
        {1200, 1, 8}
    };

    double y[] = {
        450000,
        330000,
        520000,
        280000,
        650000,
        390000,
        560000,
        240000
    };

    Linear_Regression_Model model = train_model(8, 3, x, y);
    print_model(&model);

    double x_predict[] = { 2200, 3, 21 };
    printf("Price estimated: %f\n", predict(3, x_predict, &model));
}

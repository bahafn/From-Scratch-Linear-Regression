#include <CUnit/Basic.h>

#include "../src/string_utils.h"

void test_string_to_double_integer() {
    double result = string_to_double("42");
    CU_ASSERT_DOUBLE_EQUAL(result, 42.0, 1e-12);
}

void test_string_to_double_decimal() {
    double result = string_to_double("3.14159");
    CU_ASSERT_DOUBLE_EQUAL(result, 3.14159, 1e-12);
}

void test_string_to_double_negative() {
    double result = string_to_double("-42.5");
    CU_ASSERT_DOUBLE_EQUAL(result, -42.5, 1e-12);
}

void test_string_to_double_scientific() {
    double result = string_to_double("1.5e3");
    CU_ASSERT_DOUBLE_EQUAL(result, 1500.0, 1e-12);
}

void test_string_to_double_zero() {
    double result = string_to_double("0");
    CU_ASSERT_DOUBLE_EQUAL(result, 0.0, 1e-12);
}

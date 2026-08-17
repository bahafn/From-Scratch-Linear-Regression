#pragma once

#include <ctype.h>
#include <math.h>
#include <string.h>

// The values for 10^2 to 10^20 so we don't need a long time to calculate these powers.
static const double POW10[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,
    1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20
};

static inline int char_equal_ignore_case(char a, char b) {
    if (a >= 'A' && a <= 'Z') {
        a += 'a' - 'A';
    }

    if (b >= 'A' && b <= 'Z') {
        b += 'a' - 'A';
    }

    return a == b;
}

static inline int starts_with_ignore_case(const char *str, const char *prefix) {
    while (*prefix) {
        if (*str == '\0') {
            return 0;
        }

        if (!char_equal_ignore_case(*str, *prefix)) {
            return 0;
        }

        str++;
        prefix++;
    }

    return 1;
}

static inline double string_to_double(const char *str) {
    if (!str) {
        return NAN;
    }

    const char *p = str;

    // Skip spaces
    while (isspace((unsigned char)*p)) {
        p++;
    }

    int sign = 1;
    if (*p == '-') {
        sign = -1;
        p++;
    } else if (*p == '+') {
        p++;
    }

    // Handle NaN
    if (starts_with_ignore_case(p, "nan")) {
        return NAN;
    }

    // Handle infinity
    if (starts_with_ignore_case(p, "inf")) {
        return sign * INFINITY;
    }

    double value = 0.0;
    int has_digits = 0;

    while (*p >= '0' && *p <= '9') {
        has_digits = 1;
        value = value * 10.0 + (*p - '0');
        p++;
    }

    if (*p == '.') {
        p++;

        double factor = 0.1;

        while (*p >= '0' && *p <= '9') {
            has_digits = 1;
            value += (*p - '0') * factor;
            factor *= 0.1;
            p++;
        }
    }

    // No numeric digits were found
    if (!has_digits) {
        return NAN;
    }

    if (*p == 'e' || *p == 'E') {
        p++;

        int exp_sign = 1;

        if (*p == '-') {
            exp_sign = -1;
            p++;
        } else if (*p == '+') {
            p++;
        }

        int exponent = 0;
        int has_exp_digits = 0;

        while (*p >= '0' && *p <= '9') {
            has_exp_digits = 1;
            exponent = exponent * 10 + (*p - '0');
            p++;
        }

        // "1e" or "1e+" is not a valid double
        if (!has_exp_digits) {
            return NAN;
        }

        if (exponent <= 20) {
            if (exp_sign > 0)
                value *= POW10[exponent];
            else
                value /= POW10[exponent];
        } else {
            value *= pow(10.0, exp_sign * exponent);
        }
    }

    // Reject trailing non-whitespace characters
    while (isspace((unsigned char)*p)) {
        p++;
    }

    if (*p != '\0') {
        return NAN;
    }

    return value * sign;
}

// Does the same job as strtok but doesn't consider consecutive delimeters as one delimeter
// This is used so when reading a csv file, ",," is seen as an empty element instead of being skipped
static inline char *strtok_single(char *str, char const *delims) {
    static char *src = NULL;
    char *p, *ret = 0;

    if (str != NULL) {
        src = str;
    }

    if (src == NULL) {
        return NULL;
    }

    if ((p = strpbrk(src, delims)) != NULL) {
        *p  = 0;
        ret = src;
        src = ++p;
    } else if (*src) {
        ret = src;
        src = NULL;
    }

    return ret;
}

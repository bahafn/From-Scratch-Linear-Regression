#include <stddef.h>

#define DEFAULT_DA_SIZE 10

#define BEGIN_MACRO do {
#define END_MACRO } while (0)

typedef struct {
    double *items;
    size_t  size;
    size_t  capacity;
} double_da;

#define da_init(da, init_capacity)                                          \
    BEGIN_MACRO                                                             \
        da.capacity = init_capacity;                                        \
        da.size     = 0;                                                    \
        da.items    = malloc(init_capacity * sizeof(*da.items));            \
    END_MACRO

#define da_resize(da, new_size)                                             \
    BEGIN_MACRO                                                             \
        if (da.size > new_size) da.size = new_size;                         \
        da.capacity = new_size;                                             \
        da.items    = realloc(da.items, da.capacity * sizeof(*da.items));   \
    END_MACRO

#define da_append(da, entry)                                                \
    BEGIN_MACRO                                                             \
        if (da.capacity <= da.size) {                                       \
            if (da.capacity == 0) da.capacity = DEFAULT_DA_SIZE;            \
            else da.capacity *= 1.5;                                        \
            da.items = realloc(da.items, da.capacity * sizeof(*da.items));  \
        }                                                                   \
        da.items[da.size++] = entry;                                        \
    END_MACRO

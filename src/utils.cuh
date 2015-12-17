/*
 * utils.cuh
 *
 *  Created on: 15-Dec-2015
 *      Author: saketsaurabh
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <stdint.h>

#include "data_types.h"

// For vectorizing array
typedef double (*v_func_t) (double);

bool matrix_array_allocate (matrix_array_t * const array,
                       const uint32_array_t * const dimensions);

bool vector_array_allocate (vector_array_t * const array,
                       const uint32_array_t * const dimensions,
                       const uint32_t offset);

void vector_array_zero (vector_array_t * const array);

void vector_zero (vector_t * const array);

void matrix_array_set_zero (matrix_array_t * const array);

void vector_set_rand (vector_t * const vec, const double mean, const double stddev);

void vector_array_set_rand (vector_array_t * const array,
                       const double mean,
                       const double stddev);

void matrix_set_rand (matrix_t * const mat, const double mean, const double stddev);

void matrix_array_set_rand (matrix_array_t * const array,
					   const double mean,
					   const double stddev);


void vector_array_free (vector_array_t * const array);

void matrix_array_free (matrix_array_t * const matrix);

void vector_copy(const vector_t *src, vector_t const *dest);

void vector_add (const vector_t *v1, const vector_t *v2, vector_t const *dest);

void vector_subtract (const vector_t *v1, const vector_t *v2, vector_t const *dest);

void vector_product (const vector_t *v1, const vector_t *v2, vector_t const *dest);

void matrix_vector_multiply (const matrix_t *m, const vector_t *v, vector_t const *dest);

void col_vector_multiply_row_vector_with_sum (const vector_t *col_vector, const vector_t *row_vector, matrix_t const *m);

void matrix_copy (const matrix_t *src, matrix_t *dest);

void matrix_subtract (const matrix_t *m1, const matrix_t *m2, matrix_t *dest);

void matrix_transpose (matrix_t *m);

void matrix_scale (matrix_t *m, const double factor);

void vector_scale (vector_t *v, const double factor);

void vector_vectorise (vector_t * const vec, v_func_t func);

uint32_t vector_max_index(const vector_t *v);

double uniform_random_generator (double mean, double stddev);


#endif /* UTILS_CUH_ */

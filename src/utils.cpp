
#include <stdint.h>
#include <assert.h>
#include <limits>
#include <cmath>
#include <cstdlib>
#include "utils.h"


/*
 * Allocates an array of i pointers to matrices where the dimensions of the
 * i'th matrix are: dimensions.data[i + 1] x dimensions.data[i], and i is
 * defined by dimensions.size.
 */
bool matrix_array_allocate (matrix_array_t * const array,
                       const uint32_array_t * const dimensions)
{
    array->size = dimensions->size - 1;
    array->data = new matrix_t[array->size];
    //RETURN_ERR_ON_BAD_ALLOC(array->data);

    for (uint32_t i = 0; i < array->size; ++i) {
    	array->data[i].rows = dimensions->data[i + 1];
    	array->data[i].cols = dimensions->data[i];
    	array->data[i].data = new double[array->data[i].rows * array->data[i].cols];
    }
    return true;
}

/*
 * Allocates an array of i pointers to vectors where the dimension of the i'th
 * vector is given by dimensions.data[i], and i by dimensions.size.
 *
 * Additional pointers before the 0th vector can be allocated with a finite
 * offset. This is used for negative indexing of the array.
 */
bool vector_array_allocate (vector_array_t * const array,
                       const uint32_array_t * const dimensions,
                       const uint32_t offset)
{
    array->size = dimensions->size;
    array->offset = offset;
    array->data = new vector_t[array->size + offset];
    //RETURN_ERR_ON_BAD_ALLOC(array->data);
    array->data += offset;

    for (uint32_t i = 0; i < array->size; ++i) {
        array->data[i].size = dimensions->data[i];
    	array->data[i].data = new double[array->data[i].size];
    }

    return true;
}


void vector_array_zero (vector_array_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
    	for (uint32_t j = 0; j < array->data[i].size; ++j) {
    		vector_zero(&array->data[i]);
    	}
    }
}

void vector_zero (vector_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
    	array->data[i] = 0;
    }
}

void matrix_array_set_zero (matrix_array_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
    	matrix_t m = array->data[i];
    	for (uint32_t j = 0; j < m.rows; ++j) {
    		for (uint32_t k = 0; k < m.cols; ++k) {
    			m.data[j * m.cols + k] = 0;
    		}
    	}
    }
}


void vector_set_rand (vector_t * const vec, const double mean, const double stddev)
{
    for (uint32_t i = 0; i < vec->size; ++i) {
    	vec->data[i] = uniform_random_generator(mean, stddev);
    }
}

void vector_array_set_rand (vector_array_t * const array,
                       const double mean,
                       const double stddev)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        vector_set_rand (&array->data[i], mean, stddev);
    }
}

void matrix_set_rand (matrix_t * const mat, const double mean, const double stddev)
{
	for (uint32_t i = 0; i < mat->rows; ++i) {
        for (uint32_t j = 0; j < mat->cols; ++j) {
        	mat->data[i * mat->cols + j] = uniform_random_generator(mean, stddev);
        }
    }

}

void matrix_array_set_rand (matrix_array_t * const array,
					   const double mean,
					   const double stddev)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        matrix_set_rand (&array->data[i], mean, stddev);
    }
}


void vector_array_free (vector_array_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        delete array->data[i].data;
    }
    delete (array->data - array->offset);
}

void matrix_array_free (matrix_array_t * const matrix)
{
    for (uint32_t i = 0; i < matrix->size; ++i) {
        delete matrix->data[i].data;
    }
    delete matrix->data;
}

// TODO: CUDA parallelize
void vector_vectorise (vector_t * const vec, v_func_t func)
{
    for (int i = 0; i < vec->size; ++i) {
        double tmp = (*func) (vec->data[i]);
        vec->data[i] = tmp;
    }
}

// TODO: CUDA parallelize
void vector_copy(const vector_t *src, vector_t const *dest) {
	assert(src->size == dest->size);
	for(uint32_t i = 0; i < src->size; ++i) {
		dest->data[i] = src->data[i];
	}
}


// TODO: CUDA parallelize
void vector_add (const vector_t *v1, const vector_t *v2, vector_t const *dest) {
	assert(v1->size == v2->size);
	assert(v1->size == dest->size);
	for(uint32_t i = 0; i < v1->size; ++i) {
		dest->data[i] = v1->data[i] + v2->data[i];
	}
}

// TODO: CUDA parallelize
void vector_subtract(const vector_t *v1,  const vector_t *v2, vector_t const *dest) {
	assert(v1->size == v2->size);
	assert(v1->size == dest->size);
	for(uint32_t i = 0; i < v1->size; ++i) {
		dest->data[i] = v1->data[i] - v2->data[i];
	}
}

// TODO: CUDA parallelize
void vector_product(const vector_t *v1, const vector_t *v2, vector_t const *dest) {
	assert(v1->size == v2->size);
	assert(v1->size == dest->size);
	for(uint32_t i = 0; i < v1->size; ++i) {
		dest->data[i] = v1->data[i] * v2->data[i];
	}
}

// TODO: CUDA parallelize
void matrix_vector_multiply (const matrix_t *m, const vector_t *v, vector_t const *dest) {
	assert(m->cols == v->size);
	assert(m->rows == dest->size);
	for (uint32_t i = 0; i < m->rows; ++i) {
		double accumulator = 0.0;
		for (uint32_t j = 0; j < m->cols; ++j) {
			accumulator += (m->data[i * m->cols + j] * v->data[j]);
		}
		dest->data[i] = accumulator;
	}
}

// TODO: CUDA parallelize
void col_vector_multiply_row_vector_with_sum(const vector_t *col_vector, const vector_t *row_vector, matrix_t const *m) {
	assert(m->rows == col_vector->size);
	assert(m->cols == row_vector->size);
	for (uint32_t i = 0; i < m->rows; ++i) {
		for (uint32_t j = 0; j < m->cols; ++j) {
			m->data[i * m->cols + j] += col_vector->data[i] * row_vector->data[j];
		}
	}
}

// TODO: CUDA parallelize
void matrix_copy(const matrix_t *src, matrix_t *dest) {
	assert(dest->data == NULL);
	dest->rows = src->rows;
	dest->cols = src->cols;
	dest->data = new double[dest->rows * dest->cols];
	for (uint32_t i = 0; i < dest->rows; ++i) {
		for (uint32_t j = 0; j < dest->cols; ++j) {
			dest->data[i * dest->cols + j] = src->data[i * dest->cols + j];
		}
	}
}

// TODO: CUDA parallelize
void matrix_transpose(matrix_t *m) {
	double *tmp = new double[m->rows * m->cols];
	for (uint32_t i = 0; i < m->rows; ++i) {
		for (uint32_t j = 0; j < m->cols; ++j) {
			tmp[j * m->rows + i] = m->data[i * m->cols + j];
		}
	}
	delete m->data;
	m->data = tmp;
	uint32_t tmp_val = m->rows;
	m->rows = m->cols;
	m->cols = tmp_val;
}

// TODO: CUDA parallelize
void matrix_scale (matrix_t *m, const double factor) {
	for (uint32_t i = 0; i < m->rows; ++i) {
		for (uint32_t j = 0; j < m->cols; ++j) {
			m->data[i * m->cols + j] *= factor;
		}
	}
}

// TODO: CUDA parallelize
void vector_scale (vector_t *v, const double factor) {
	for(uint32_t i = 0; i < v->size; ++i) {
		v->data[i] *= factor;
	}
}

// TODO: CUDA parallelize
void matrix_subtract (const matrix_t *m1, const matrix_t *m2, matrix_t *dest) {
	assert(m1->rows == m2->rows && m1->rows == dest->rows);
	assert(m1->cols == m2->cols && m1->cols == dest->cols);

	for (uint32_t i = 0; i < dest->rows; ++i) {
		for (uint32_t j = 0; j < dest->cols; ++j) {
			dest->data[i * dest->cols + j] = m1->data[i * dest->cols + j] - m2->data[i * dest->cols + j];
		}
	}
}


uint32_t vector_max_index(const vector_t *v) {
	double max = std::numeric_limits<double>::min();
	uint32_t maxIdx = 0;
	for(uint32_t i = 0; i < v->size; ++i) {
		if (v->data[i] > max) {
			max = v->data[i];
			maxIdx = i;
		}
	}
	return maxIdx;
}

double uniform_random_generator (double mean, double stddev)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mean + stddev * (double) X2);
    }

  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mean + stddev * (double) X1);
}



/**
 * Filename: data_types.h
 * Authors: Saket Saurabh, Shashank Gupta
 * Language: C++
 * To Compile: Please check README.txt
 * Description: Defines the various data structures used for representing the
 * 				dataset and the neural network.
 */


#ifndef DATA_TYPES_H_
#define DATA_TYPES_H_

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#define IMAGES_HEADER_SIZE_BYTES 16
#define LABELS_HEADER_SIZE_BYTES 8

typedef struct
{
    int32_t magic_num;
    int32_t num_images;
    int32_t rows;
    int32_t cols;
    int32_t pixels;
    double* images;
} images_t;

typedef struct
{
    int32_t magic_num;
    int32_t num_labels;
    uint8_t* labels;
} labels_t;

typedef struct
{
    labels_t labels;
    images_t images;
    uint32_t items;
} data_t;


typedef struct
{
	uint32_t rows;
	uint32_t cols;
	double* data;
} matrix_t;

typedef struct
{
	uint32_t size;
	double* data;
} vector_t;


typedef struct
{
    uint32_t size;
    uint32_t* data;
} uint32_array_t;

typedef struct
{
    uint32_t size;
    uint32_t offset;
    vector_t* data;
} vector_array_t;

typedef struct
{
    uint32_t size;
    matrix_t* data;
} matrix_array_t;


typedef struct
{
    double eta;
    uint32_t epochs;
    uint32_t mini_batch_size;
    uint32_array_t nodes;
    vector_t inputs;
    vector_array_t outputs;
    vector_array_t zs;
    vector_array_t nabla_b;
    vector_array_t output_delta;
    vector_array_t biases;
    matrix_array_t weights;
    matrix_array_t nabla_w;
} network_t;


typedef struct
{
	uint32_t num_vectors;
    uint32_t size;
    uint32_t* size_array;
    uint32_t* offset_positions;
    double* data;
} device_vector_array_t;

typedef struct
{
	uint32_t num_matrices;
    uint32_t size;
    uint32_t* offset_positions;
    uint32_t* rows_array;
    uint32_t* cols_array;
    double* data;
} device_matrix_array_t;


typedef struct
{
    double eta;
    uint32_t epochs;
    uint32_t mini_batch_size;
    uint32_array_t nodes;
    vector_t inputs;
    device_vector_array_t outputs;
    device_vector_array_t zs;
    device_vector_array_t nabla_b;
    device_vector_array_t output_delta;
    device_vector_array_t biases;
    device_matrix_array_t weights;
    device_matrix_array_t nabla_w;
    vector_t store1;
    vector_t store2;
} device_network_t;



#endif /* DATA_TYPES_H_ */

/**
 * Filename: neural_network.cuh
 * Authors: Saket Saurabh, Shashank Gupta
 * Language: C++
 * To Compile: Please check README.txt
 * Description: The corresponding header file for neural_network.cu
 */

#ifndef NEURAL_NETWORK_CUH_
#define NEURAL_NETWORK_CUH_

#include <math.h>
#include <stdint.h>

#include "data_loader.h"
#include "data_types.h"
#include "utils.h"



typedef void (*update_batch_f) (network_t * const net,
							const data_t data_d,
							const uint32_array_t rand_index_d,
							const uint32_t beginIndex,
							const uint32_t endIndex);

void network_free (network_t * const network);


bool network_allocate (network_t * const network);

void network_random_init (network_t * const network, const double mean, const double stddev);

void network_sgd (network_t * const network,
             const data_t * const data,
             const data_t * const test_data);

void network_process_mini_batches (network_t * const net,
                              const data_t data_d,
                              const uint32_array_t rand_index_d,
                              update_batch_f update_batch);

void network_update_mini_batch (network_t * const net,
                           const data_t data_d,
                           const uint32_array_t rand_index_d,
                           const uint32_t beginIndex,
                           const uint32_t endIndex);

void network_get_output_error (network_t * const network, const uint8_t label);

void network_accumulate_cfgs (network_t * const network, const int32_t layer);

void network_backpropagate_error (network_t * const network, const uint8_t label);

void network_evaluate_test_data (network_t * const network,
                            const data_t * const test_data,
                            uint32_t * const correct_answers);

void network_evaluate_output (network_t * const network, uint32_t * const output);

void network_get_output (network_t * const net, uint32_t * const output);

void network_feed_forward (const network_t * const network, const uint8_t store_z);

double sigmoid (double z);

double sigmoid_prime (double z);

void cost_derivative (const vector_t * output_activations,
                 const vector_t * expected_output_y,
                 vector_t * const cost_derivative);


data_t allocate_device_image_data(const data_t* const data_h);

void copy_to_device_image_data(data_t data_d, const data_t* const data_h);

void free_device_image_data(data_t data_d);

uint32_array_t allocate_device_array(const uint32_array_t* const array_h);

void copy_to_device_array(uint32_array_t array_d, const uint32_array_t* const array_h);

void free_device_array(uint32_array_t array_d);

device_network_t copy_to_device_network(const network_t* const net_h);

void copy_device_vector_array_helper(device_vector_array_t *array_d, const vector_array_t* array_h);

void copy_device_matrix_array_helper(device_matrix_array_t *array_d, const matrix_array_t *array_h);

void free_device_network(device_network_t net_d);

void copy_nabla_from_device(device_network_t net_d, matrix_array_t* nabla_w, vector_array_t* nabla_b);

#endif /* NEURAL_NETWORK_CUH_ */

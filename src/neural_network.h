
#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_



#include <math.h>
#include <stdint.h>

#include "data_loader.h"
#include "utils.h"

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

typedef void (*update_batch_f) (network_t * const,
                   const data_t * const,
                   const uint32_array_t * const);

void network_free (network_t * const network);


bool network_allocate (network_t * const network);

void network_random_init (network_t * const network, const double mean, const double stddev);

void network_sgd (network_t * const network,
             const data_t * const data,
             const data_t * const test_data);

void network_process_mini_batches (network_t * const network,
                              const data_t * const data,
                              const uint32_t * const rnd_idx,
                              update_batch_f update_batch_f);

void network_update_mini_batch (network_t * const network,
                           const data_t * const data,
                           const uint32_array_t * const array_slice);

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

#endif


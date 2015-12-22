/**
 * Filename: neural_network.cpp
 * Authors: Saket Saurabh, Shashank Gupta
 * Language: C++
 * To Compile: Please check README.txt
 * Description: The sequential implementation of the neural network for
 * 				stochastic gradient descent. Contains the main steps of the
 * 				algorithm broken down into various functions.
 */

#include <assert.h>
#include <algorithm>
#include <unistd.h>

#include "neural_network.h"
#include "utils.h"


bool network_allocate (network_t * const net)
{
	bool err;

	matrix_array_allocate (&net->weights, &net->nodes);
    matrix_array_allocate (&net->nabla_w, &net->nodes);

    uint32_array_t dimensions;
    dimensions.size = net->nodes.size - 1;
    dimensions.data = net->nodes.data + 1;

    vector_array_allocate (&net->zs, &dimensions, 0);
    vector_array_allocate (&net->nabla_b, &dimensions, 0);
    vector_array_allocate (&net->output_delta, &dimensions, 0);
    vector_array_allocate (&net->biases, &dimensions, 0);

    vector_array_allocate (&net->outputs, &dimensions, 0);

    // allocate input vector
    net->inputs.size = net->nodes.data[0];
    net->inputs.data = new double[net->inputs.size];

    return err;
}


void network_random_init (network_t * const network, const double mean, const double stddev) {
	vector_array_set_rand (&network->biases, mean, stddev);
	matrix_array_set_rand (&network->weights, mean, stddev);
}


void network_free (network_t * const net)
{
    vector_array_free (&net->outputs);
    vector_array_free (&net->zs);
    vector_array_free (&net->nabla_b);
    vector_array_free (&net->output_delta);
    vector_array_free (&net->biases);

    matrix_array_free (&net->nabla_w);
    matrix_array_free (&net->weights);
}

/*
 * 	Stochastic Gradient Descent
 */
void network_sgd (network_t * const net,
             const data_t * const data,
             const data_t * const test_data)
{

    // Index array used to address labels and images in random order
    uint32_t rand_index[data->items];
    for (uint32_t i = 0; i < data->items; ++i) {
        rand_index[i] = i;
    }

    for (uint32_t i = 0; i < net->epochs; ++i)
    {
    	// Randomize the index array
    	std::random_shuffle(rand_index, rand_index + (sizeof(rand_index) / sizeof(rand_index[0])));

    	network_process_mini_batches (net, data, &rand_index[0], &network_update_mini_batch);

        uint32_t correct_answers = 0;
        network_evaluate_test_data (net, test_data, &correct_answers);

        printf ("Epoch %i complete, %i/%i correct.\n", i, correct_answers,
                test_data->items);
    }
}

void print_vector(vector_t *v) {
	for(uint32_t i = 0; i < v->size; i++) {
		printf("(%d,%f);",i,v->data[i]);
	}
	printf("\n");
}
void print_matrix(matrix_t *m) {
	for(uint32_t i = 0; i < m->rows * m->cols; i++) {
			printf("(%d,%f),",i,m->data[i]);
		}
		printf("\n");
}



void network_process_mini_batches (network_t * const net,
                              const data_t * const data,
                              const uint32_t * const rand_index,
                              update_batch_f update_batch)
{
    assert(data->items != 0);
    assert(net->mini_batch_size != 0);

    // A slice of randomized indexes for the input data
    uint32_array_t slice;
    slice.data = (uint32_t *) rand_index;
    slice.size = net->mini_batch_size;

    uint32_t batches = data->items / net->mini_batch_size;
    printf ("Iterating over %i batches...\n", batches);

    for (uint32_t i = 0; i < batches; ++i) {
        update_batch (net, data, &slice);
        slice.data += slice.size;
    }

    uint32_t remainder = data->items % net->mini_batch_size;
    if (remainder) {
        slice.size = remainder;
        update_batch (net, data, &slice);
    }
}


void network_update_mini_batch (network_t * const net,
                           const data_t * const data,
                           const uint32_array_t * const slice)
{
    assert(slice->size != 0);

    // Reset batch averages
    vector_array_zero (&net->nabla_b);
    matrix_array_set_zero (&net->nabla_w);

    // Apply SGD to the mini-batch
    for (uint32_t i = 0; i < slice->size; ++i) {
        uint32_t random_index = slice->data[i];
        // load image for the given random_index
        assert(net->inputs.size == data->images.pixels);
        for (uint32_t j = 0; j < data->images.pixels; ++j) {
        	net->inputs.data[j] = data->images.images[random_index * data->images.pixels + j];
        }
        network_backpropagate_error (net, data->labels.labels[random_index]);
     }

    // Update weights and biases
    double scale_fac = net->eta / slice->size;


    uint32_t whole_layers = net->nodes.size - 1;
    for (uint32_t i = 0; i < whole_layers; ++i) {
        matrix_scale (&net->nabla_w.data[i], scale_fac);
        matrix_subtract (&net->weights.data[i], &net->nabla_w.data[i], &net->weights.data[i]);

        vector_scale (&net->nabla_b.data[i], scale_fac);
        vector_subtract (&net->biases.data[i], &net->nabla_b.data[i], &net->biases.data[i]);
    }

}


void network_backpropagate_error (network_t * const net, const uint8_t label)
{
    network_feed_forward (net, 1);
    network_get_output_error (net, label);

    uint32_t output_layer_index = net->outputs.size - 1;
    network_accumulate_cfgs (net, output_layer_index);

    // Back propagate
    for (int32_t l = output_layer_index - 1; l >= 0; --l)
    {
        vector_copy (&net->zs.data[l], &net->output_delta.data[l]);

        vector_vectorise (&net->output_delta.data[l], &sigmoid_prime);

        vector_t tmp_vector;
        tmp_vector.size = (net->output_delta.data[l].size);
        tmp_vector.data = new double[tmp_vector.size];

        // create a copy of weights.data[l+1] matrix and then transpose
        matrix_t tmp_matrix;
        tmp_matrix.data = NULL;
        matrix_copy(&net->weights.data[l+1], &tmp_matrix); // matrix_copy(src, dest)
        matrix_transpose(&tmp_matrix);


        // Y = alpha(A^T) + beta(Y)
        matrix_vector_multiply (&tmp_matrix, &net->output_delta.data[l + 1], &tmp_vector);

        // Back-propagated delta
        vector_product (&tmp_vector, &net->output_delta.data[l], &net->output_delta.data[l]);

        network_accumulate_cfgs (net, l);

        delete tmp_vector.data;
        delete tmp_matrix.data;
    }
}


/*
 * Calculate the output vector from the input
 *
 */
void network_feed_forward (const network_t * const net, const uint8_t store_z)
{
    uint32_t whole_layers = net->nodes.size - 1;

    // use input layer to compute activation values for first hidden layer
    // a^0 = w^0 * input + b^0
    matrix_vector_multiply (&net->weights.data[0], &net->inputs, &net->outputs.data[0]);
    vector_add (&net->outputs.data[0], &net->biases.data[0], &net->outputs.data[0]);
    if (store_z) {
        vector_copy (&net->outputs.data[0], &net->zs.data[0]); // vector_copy(src, dest)
    }
    vector_vectorise (&net->outputs.data[0], &sigmoid);

    for (int32_t i = 1; i < whole_layers; ++i)
    {
        // a^l = w^l * a^(l-1) + b^l
    	matrix_vector_multiply (&net->weights.data[i], &net->outputs.data[i - 1], &net->outputs.data[i]);
    	vector_add (&net->outputs.data[i], &net->biases.data[i], &net->outputs.data[i]);

        if (store_z) {
            vector_copy (&net->outputs.data[i], &net->zs.data[i]); // vector_copy(src, dest)
        }

        vector_vectorise (&net->outputs.data[i], &sigmoid);
    }

}


void network_get_output_error (network_t * const net, const uint8_t label)
{
    uint32_t output_layer_index = net->outputs.size - 1;

    // create expected output vector y from label
    vector_t expected_output_y;
    expected_output_y.size = net->outputs.data[output_layer_index].size;
    expected_output_y.data = new double[expected_output_y.size];
    vector_zero(&expected_output_y);
    expected_output_y.data[label] = 1;

    // create cost_derivative vector
    vector_t cost_deriv;
    cost_deriv.size = net->outputs.data[output_layer_index].size;
    cost_deriv.data = new double[cost_deriv.size];

    cost_derivative (&net->outputs.data[output_layer_index], &expected_output_y, &cost_deriv);

    vector_copy (&net->zs.data[output_layer_index], &net->output_delta.data[output_layer_index]); // vector_copy(src, dest)

    vector_vectorise (&net->output_delta.data[output_layer_index],
                      &sigmoid_prime);

    vector_product(&net->output_delta.data[output_layer_index], &cost_deriv, &net->output_delta.data[output_layer_index]);

    // free memory
    delete cost_deriv.data;
    delete expected_output_y.data;

}

/*
 * Accumulate cost function gradients
 */
void network_accumulate_cfgs (network_t * const net, const int32_t layer)
{
    // Y = alphaX + Y
	vector_add (&net->output_delta.data[layer], &net->nabla_b.data[layer], &net->nabla_b.data[layer]);

    // A = [delta] * [activations]^T + A
	if (layer >= 1) {
		col_vector_multiply_row_vector_with_sum(&net->output_delta.data[layer], &net->outputs.data[layer - 1], &net->nabla_w.data[layer]);
	} else {
		col_vector_multiply_row_vector_with_sum(&net->output_delta.data[layer], &net->inputs, &net->nabla_w.data[layer]);
	}

}

void network_evaluate_test_data (network_t * const net,
                            const data_t * const test_data,
                            uint32_t * const correct_answers)
{
    *correct_answers = 0;
    uint32_t output;

    for (uint32_t i = 0; i < test_data->items; ++i) {

    	// load test image for the given index i
		assert(net->inputs.size == test_data->images.pixels);
		for (uint32_t j = 0; j < test_data->images.pixels; ++j) {
			net->inputs.data[j] = test_data->images.images[i * test_data->images.pixels + j];
		}

        network_get_output (net, &output);
        if (output == test_data->labels.labels[i])
            (*correct_answers)++;
    }
}

void network_get_output (network_t * const net, uint32_t * const output)
{
    network_feed_forward (net, 0);

    // Returns the lowest index if more than 1.
    *output = vector_max_index (&net->outputs.data[net->outputs.size - 1]);
}



double sigmoid (double z)
{
    return 1.0 / (1.0 + exp (-z));
}

double sigmoid_prime (double z)
{
    double sig_z = sigmoid (z);
    return sig_z * (1.0 - sig_z);
}


void cost_derivative (const vector_t * output_activations,
                 const vector_t * expected_output_y,
                 vector_t * const cost_derivative)
{
    // Subtract expected output (unit vector) from the output
    vector_subtract(output_activations, expected_output_y, cost_derivative);

}



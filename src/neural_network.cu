/*
 * neural_network.cu
 *
 *  Created on: 15-Dec-2015
 *      Author: saketsaurabh
 */

#include <assert.h>
#include <algorithm>    // std::shuffle
#include <unistd.h>
#include <cuda.h>

#include "neural_network.cuh"
#include "utils.cuh"
#include "backpropagation.cuh"


bool network_allocate (network_t * const net)
{
    //err_t err = GSL_SUCCESS;
	bool err = true;

	matrix_array_allocate (&net->weights, &net->nodes);
    matrix_array_allocate (&net->nabla_w, &net->nodes);

    // These arrays are not required for the input layer
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
             const data_t * const data_h,
             const data_t * const test_data)
{

    // Index array used to address labels and images in random order
    uint32_array_t *rand_index_h = new uint32_array_t;
    rand_index_h->size = data_h->items;
    rand_index_h->data = new uint32_t[rand_index_h->size];
    for (uint32_t i = 0; i < rand_index_h->size; ++i) {
    	rand_index_h->data[i] = i;
    }

    // allocate and copy images to device
    data_t data_d = allocate_device_image_data(data_h);
    copy_to_device_image_data(data_d, data_h);

    // allocate rand_index_d for device
     uint32_array_t rand_index_d = allocate_device_array(rand_index_h);


    for (uint32_t i = 0; i < net->epochs; ++i)
    {
    	// Randomize the index array
    	//std::random_shuffle(rand_index_h->data,rand_index_h->data + (sizeof(rand_index_h->data) / sizeof(rand_index_h->data[0])));

    	copy_to_device_array(rand_index_d, rand_index_h);

    	network_process_mini_batches (net, data_d, rand_index_d, &network_update_mini_batch);

        uint32_t correct_answers = 0;
        network_evaluate_test_data (net, test_data, &correct_answers);

        printf ("Epoch %i complete, %i/%i correct.\n", i, correct_answers,
                test_data->items);
    }

    free_device_image_data(data_d);
    free_device_array(rand_index_d);
}

void print_vector(vector_t *v) {
	for(uint32_t i = 0; i < v->size; i++) {
		printf("%f\t",v->data[i]);
	}
	printf("\n");
}


void network_process_mini_batches (network_t * const net,
                              const data_t data_d,
                              const uint32_array_t rand_index_d,
                              update_batch_f update_batch)
{
    assert(data_d.items != 0);
    assert(net->mini_batch_size != 0);

    uint32_t batches = data_d.items / net->mini_batch_size;
    printf ("Iterating over %i batches...\n", batches);
    uint32_t iter = 0;
    for (uint32_t i = 0; i < data_d.items; i += net->mini_batch_size) {
    	uint32_t beginIndex = i;
    	uint32_t endIndex = i + net->mini_batch_size - 1;
    	endIndex = (endIndex < data_d.items) ? endIndex : data_d.items - 1; // out-of-bounds check
    	update_batch (net, data_d, rand_index_d, beginIndex, endIndex);
    	iter++;
    }

    printf("No. of iterations: %u\n" , iter);

}


void network_update_mini_batch (network_t * const net_h,
                           const data_t data_d,
                           const uint32_array_t rand_index_d,
                           const uint32_t beginIndex,
                           const uint32_t endIndex)
{
    assert(beginIndex <= endIndex);
    assert(endIndex - beginIndex + 1 == net_h->mini_batch_size);

    // Reset batch averages
    vector_array_zero (&net_h->nabla_b);
    matrix_array_set_zero (&net_h->nabla_w);

    device_network_t net_d = copy_to_device_network(net_h);


    /*// call backpropagation kernel here
    // execution configuration: <endIndex - beginIndex + 1, 1024>
    for (uint32_t i = beginIndex; i <= endIndex; ++i) {
    	backpropagation_kernel<<<1,1024>>>(net_d, data_d, rand_index_d, i, endIndex);
    	cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) {
        	printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
    }*/

    for (uint32_t i = beginIndex; i <= endIndex; i++) {
    	backpropagation_kernel<<<1,1024>>>(net_d, data_d, rand_index_d, i, endIndex);
    	copy_nabla_from_device(net_d, &net_h->nabla_w, &net_h->nabla_b);
    }

    free_device_network(net_d);


    // Update weights and biases
    double scale_fac = net_h->eta / net_h->mini_batch_size;


    uint32_t whole_layers = net_h->nodes.size - 1;
    for (uint32_t i = 0; i < whole_layers; ++i) {
        matrix_scale (&net_h->nabla_w.data[i], scale_fac);
        matrix_subtract (&net_h->weights.data[i], &net_h->nabla_w.data[i], &net_h->weights.data[i]);

        vector_scale (&net_h->nabla_b.data[i], scale_fac);
        vector_subtract (&net_h->biases.data[i], &net_h->nabla_b.data[i], &net_h->biases.data[i]);
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


data_t allocate_device_image_data(const data_t* const data_h) {
	uint32_t size;
	data_t data_d;

	// allocate labels data
	data_d.labels.magic_num = data_h->labels.magic_num;
	data_d.labels.num_labels = data_h->labels.num_labels;
	size = data_d.labels.num_labels * sizeof(uint8_t);
	cudaMalloc((void**) &data_d.labels.labels, size);

	// allocate images data
	data_d.images.magic_num = data_h->images.magic_num;
	data_d.images.num_images = data_h->images.num_images;
	data_d.images.rows = data_h->images.rows;
	data_d.images.cols = data_h->images.cols;
	data_d.images.pixels = data_h->images.pixels;
	size = data_d.images.num_images * data_d.images.pixels * sizeof(double);
	cudaMalloc((void**) &data_d.images.images, size);

	// allocate items data
	data_d.items = data_h->items;

	return data_d;
}

void copy_to_device_image_data(data_t data_d, const data_t* const data_h) {
	uint32_t size;
	// copy labels data
	size = data_d.labels.num_labels * sizeof(uint8_t);
	cudaMemcpy(data_d.labels.labels, data_h->labels.labels, size, cudaMemcpyHostToDevice);

	// copy images data
	size = data_d.images.num_images * data_d.images.pixels * sizeof(double);
	cudaMemcpy(data_d.images.images, data_h->images.images, size, cudaMemcpyHostToDevice);

}

void free_device_image_data(data_t data_d) {
	cudaFree(data_d.labels.labels);
	cudaFree(data_d.images.images);
}

uint32_array_t allocate_device_array(const uint32_array_t* const array_h) {
	uint32_array_t array_d;
	array_d.size = array_h->size;
	cudaMalloc((void**) &array_d.data, array_d.size * sizeof(uint32_t));
	return array_d;
}

void copy_to_device_array(uint32_array_t array_d, const uint32_array_t* const array_h) {
	cudaMemcpy(array_d.data, array_h->data, array_d.size * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void free_device_array(uint32_array_t array_d) {
	cudaFree(array_d.data);
}


device_network_t copy_to_device_network(const network_t* const net_h) {
	device_network_t net_d;
	uint32_t size;

	// copy constants
	net_d.eta = net_h->eta;
	net_d.epochs = net_h->epochs;
	net_d.mini_batch_size = net_h->mini_batch_size;

	// copy nodes
	net_d.nodes.size = net_h->nodes.size;
	size = net_d.nodes.size * sizeof(uint32_t);
	cudaMalloc((void**) &net_d.nodes.data, size);
	cudaMemcpy(net_d.nodes.data, net_h->nodes.data, size, cudaMemcpyHostToDevice);

	// copy input vector
	net_d.inputs.size = net_h->inputs.size;
	cudaMalloc((void**) &net_d.inputs.data, net_d.inputs.size * sizeof(double));
	cudaMemcpy(net_d.inputs.data, net_h->inputs.data, net_d.inputs.size * sizeof(double), cudaMemcpyHostToDevice);

	// copy vector arrays
	copy_device_vector_array_helper(&net_d.outputs, &net_h->outputs);
	copy_device_vector_array_helper(&net_d.zs, &net_h->zs);
	copy_device_vector_array_helper(&net_d.nabla_b, &net_h->nabla_b);
	copy_device_vector_array_helper(&net_d.output_delta, &net_h->output_delta);
	copy_device_vector_array_helper(&net_d.biases, &net_h->biases);

	// copy matrix arrays
	copy_device_matrix_array_helper(&net_d.weights, &net_h->weights);
	copy_device_matrix_array_helper(&net_d.nabla_w, &net_h->nabla_w);

	// allocate temporary stores in global memory
	net_d.store1.size = 1000;
	net_d.store2.size = 50000;
	cudaMalloc((void**) &net_d.store1.data, net_d.store1.size * sizeof(double));
	cudaMalloc((void**) &net_d.store2.data, net_d.store2.size * sizeof(double));

	return net_d;
}

void copy_device_vector_array_helper(device_vector_array_t *array_d, const vector_array_t* array_h) {
	device_vector_array_t tmp;
	tmp.num_vectors = array_h->size;
	tmp.offset_positions = new uint32_t[tmp.num_vectors];
	tmp.size_array = new uint32_t[tmp.num_vectors];

	// determine the size of the tmp's data
	tmp.size = 0;
	for (uint32_t i = 0; i < array_h->size; ++i) {
		tmp.size_array[i] = array_h->data[i].size;
		tmp.size += array_h->data[i].size;
	}

	tmp.data = new double[tmp.size];


	uint32_t k = 0;
	for (uint32_t i = 0; i < array_h->size; ++i) {
		tmp.offset_positions[i] = k;
		for (uint32_t j = 0; j < array_h->data[i].size; ++j) {
			tmp.data[k] = array_h->data[i].data[j];
			++k;
		}
	}

	// copy tmp to array_d
	array_d->num_vectors = tmp.num_vectors;
	array_d->size = tmp.size;
	cudaMalloc((void**) &array_d->offset_positions, array_d->num_vectors * sizeof(uint32_t));
	cudaMemcpy(array_d->offset_positions, tmp.offset_positions,
			array_d->num_vectors * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &array_d->size_array, array_d->num_vectors * sizeof(uint32_t));
	cudaMemcpy(array_d->size_array, tmp.size_array,
				array_d->num_vectors * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &array_d->data, array_d->size * sizeof(double));
	cudaMemcpy(array_d->data, tmp.data, array_d->size * sizeof(double), cudaMemcpyHostToDevice);

	// free tmp
	delete tmp.offset_positions;
	delete tmp.size_array;
	delete tmp.data;
}

void copy_device_matrix_array_helper(device_matrix_array_t *array_d, const matrix_array_t *array_h) {
	device_matrix_array_t tmp;
	tmp.num_matrices = array_h->size;
	tmp.offset_positions = new uint32_t[tmp.num_matrices];
	tmp.rows_array = new uint32_t[tmp.num_matrices];
	tmp.cols_array = new uint32_t[tmp.num_matrices];

	// determine the size of the tmp's data
	tmp.size = 0;
	for (uint32_t i = 0; i < array_h->size; ++i) {
		tmp.rows_array[i] = array_h->data[i].rows;
		tmp.cols_array[i] = array_h->data[i].cols;
		tmp.size +=  tmp.rows_array[i] * tmp.cols_array[i];
	}
	tmp.data = new double[tmp.size];

	uint32_t k = 0;
	for (uint32_t i = 0; i < array_h->size; ++i) {
		tmp.offset_positions[i] = k;
		uint32_t size_m = tmp.rows_array[i] * tmp.cols_array[i];
		for (uint32_t j = 0; j < size_m; ++j) {
			tmp.data[k] = array_h->data[i].data[j];
			++k;
		}
	}

	// copy tmp to array_d
	array_d->num_matrices = tmp.num_matrices;
	array_d->size = tmp.size;
	uint32_t size = array_d->num_matrices * sizeof(uint32_t);
	cudaMalloc((void**) &array_d->offset_positions, size);
	cudaMemcpy(array_d->offset_positions, tmp.offset_positions, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &array_d->rows_array, size);
	cudaMemcpy(array_d->rows_array, tmp.rows_array, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &array_d->cols_array, size);
	cudaMemcpy(array_d->cols_array, tmp.cols_array, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &array_d->data, array_d->size * sizeof(double));
	cudaMemcpy(array_d->data, tmp.data, array_d->size * sizeof(double), cudaMemcpyHostToDevice);


	// free tmp
	delete tmp.offset_positions;
	delete tmp.rows_array;
	delete tmp.cols_array;
	delete tmp.data;
}

void copy_nabla_from_device(device_network_t net_d, matrix_array_t* nabla_w, vector_array_t* nabla_b) {

	// copy to nabla_w
	device_matrix_array_t nabla_w_tmp;

	nabla_w_tmp.num_matrices = net_d.nabla_w.num_matrices;
	nabla_w_tmp.size = net_d.nabla_w.size;
	nabla_w_tmp.offset_positions = new uint32_t[nabla_w_tmp.num_matrices];
	nabla_w_tmp.rows_array = new uint32_t[nabla_w_tmp.num_matrices];
	nabla_w_tmp.cols_array = new uint32_t[nabla_w_tmp.num_matrices];
	nabla_w_tmp.data = new double[nabla_w_tmp.size];

	uint32_t size = nabla_w_tmp.num_matrices * sizeof(uint32_t);
	cudaMemcpy(nabla_w_tmp.offset_positions, net_d.nabla_w.offset_positions, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nabla_w_tmp.rows_array, net_d.nabla_w.rows_array, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nabla_w_tmp.cols_array, net_d.nabla_w.cols_array, size, cudaMemcpyDeviceToHost);

	size = nabla_w_tmp.size * sizeof(double);
	cudaMemcpy(nabla_w_tmp.data, net_d.nabla_w.data, size, cudaMemcpyDeviceToHost);

	uint32_t pos = 0;
	for (uint32_t k = 0; k < nabla_w_tmp.num_matrices; ++k) {
		uint32_t rows = nabla_w_tmp.rows_array[k];
		uint32_t cols = nabla_w_tmp.cols_array[k];
		for(uint32_t i = 0; i < rows; i++) {
			for(uint32_t j = 0; j < cols; ++j) {
				nabla_w->data[k].data[i * cols + j] = nabla_w_tmp.data[pos];
				pos++;
			}
		}
	}

	delete nabla_w_tmp.offset_positions;
	delete nabla_w_tmp.rows_array;
	delete nabla_w_tmp.cols_array;
	delete nabla_w_tmp.data;

	// copy to nabla_b
	device_vector_array_t nabla_b_tmp;
	nabla_b_tmp.num_vectors = net_d.nabla_b.num_vectors;
	nabla_b_tmp.size = net_d.nabla_b.size;
	nabla_b_tmp.size_array = new uint32_t[nabla_b_tmp.num_vectors];
	nabla_b_tmp.offset_positions = new uint32_t[nabla_b_tmp.num_vectors];
	nabla_b_tmp.data = new double[nabla_b_tmp.size];

	size = nabla_b_tmp.num_vectors * sizeof(uint32_t);
	cudaMemcpy(nabla_b_tmp.offset_positions, net_d.nabla_b.offset_positions, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nabla_b_tmp.size_array, net_d.nabla_b.size_array, size, cudaMemcpyDeviceToHost);
	size = nabla_b_tmp.size * sizeof(double);
	cudaMemcpy(nabla_b_tmp.data, net_d.nabla_b.data, size, cudaMemcpyDeviceToHost);

	pos = 0;
	for (uint32_t k = 0; k < nabla_b_tmp.num_vectors; ++k) {
		for (uint32_t i = 0; i < nabla_b->data[k].size; ++i) {
			nabla_b->data[k].data[i] = nabla_b_tmp.data[pos];
			pos++;
		}
	}
	delete nabla_b_tmp.offset_positions;
	delete nabla_b_tmp.size_array;
	delete nabla_b_tmp.data;


}

void free_device_network(device_network_t net_d) {
	cudaFree(net_d.nodes.data);
	cudaFree(net_d.inputs.data);
	cudaFree(net_d.outputs.data);
	cudaFree(net_d.outputs.offset_positions);
	cudaFree(net_d.outputs.size_array);
	cudaFree(net_d.zs.data);
	cudaFree(net_d.zs.offset_positions);
	cudaFree(net_d.zs.size_array);
	cudaFree(net_d.nabla_b.data);
	cudaFree(net_d.nabla_b.offset_positions);
	cudaFree(net_d.nabla_b.size_array);
	cudaFree(net_d.output_delta.data);
	cudaFree(net_d.output_delta.offset_positions);
	cudaFree(net_d.output_delta.size_array);
	cudaFree(net_d.biases.data);
	cudaFree(net_d.biases.offset_positions);
	cudaFree(net_d.biases.size_array);
	cudaFree(net_d.weights.data);
	cudaFree(net_d.weights.offset_positions);
	cudaFree(net_d.weights.rows_array);
	cudaFree(net_d.weights.cols_array);
	cudaFree(net_d.nabla_w.data);
	cudaFree(net_d.nabla_w.offset_positions);
	cudaFree(net_d.nabla_w.rows_array);
	cudaFree(net_d.nabla_w.cols_array);
	cudaFree(net_d.store1.data);
	cudaFree(net_d.store2.data);
}


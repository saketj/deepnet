#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "neural_network.cuh"
#include "utils.cuh"
#include "backpropagation.cuh"

__device__ double device_sigmoid (double z)
{
    return 1.0 / (1.0 + exp (-z));
}

__device__ double device_sigmoid_prime (double z)
{
    double sig_z = device_sigmoid (z);
    return sig_z * (1.0 - sig_z);
}


__global__ void backpropagation_kernel(device_network_t net,
									const data_t data,
									const uint32_array_t rand_index,
									const uint32_t beginIndex,
									const uint32_t endIndex) {
	uint32_t block_id = blockIdx.x;
	uint32_t thread_id = threadIdx.x;
	uint32_t num_threads = blockDim.x;

	uint32_t image_id = rand_index.data[block_id + beginIndex];
	uint8_t label = data.labels.labels[image_id];


	// step 1: load image data
	if (thread_id < net.inputs.size) {
		net.inputs.data[thread_id] = data.images.images[image_id * data.images.pixels + thread_id];
	}
	__syncthreads();


	// step 2: network feed forward
	//matrix_vector_multiply (&net.weights.data[0], &net.inputs, &net.outputs.data[0]);
	if (thread_id < net.weights.rows_array[0]) { // thread_id == row_index
		double accumulator = 0.0;
		uint32_t cols = net.weights.cols_array[0];
		for (uint32_t i = 0; i < cols; ++i) { // i == col_index
			double m = net.weights.data[net.weights.offset_positions[0] + thread_id * cols + i];
			double n = net.inputs.data[i];
			accumulator += m * n;
		}
		uint32_t pos = net.outputs.offset_positions[0] + thread_id;
		net.outputs.data[pos] = accumulator;
	}
	__syncthreads();
	//vector_add (&net.outputs.data[0], &net.biases.data[0], &net.outputs.data[0]);
	if (thread_id < (net.outputs.size_array[0]) ) {
		uint32_t pos = net.outputs.offset_positions[0] + thread_id;
		net.outputs.data[pos] += net.biases.data[pos];
	}
	__syncthreads();
	//vector_copy (&net->outputs.data[0], &net->zs.data[0]); // vector_copy(src, dest)
	if (thread_id < (net.outputs.size_array[0]) ) {
		uint32_t pos = net.outputs.offset_positions[0] + thread_id;
		net.zs.data[pos] = net.outputs.data[pos];
	}
	__syncthreads();
	//vector_vectorise (&net->outputs.data[0], &sigmoid);
	if (thread_id < (net.outputs.size_array[0]) ) {
		uint32_t pos = net.outputs.offset_positions[0] + thread_id;
		net.outputs.data[pos] = device_sigmoid(net.outputs.data[pos]);
	}
	__syncthreads();
	uint32_t whole_layers = net.nodes.size - 1;
	for (uint32_t k = 1; k < whole_layers; ++k)
	{
		//matrix_vector_multiply (&net->weights.data[i], &net->outputs.data[i - 1], &net->outputs.data[i]);
		if (thread_id < net.weights.rows_array[k]) { // thread_id == row_index
			double accumulator = 0.0;
			uint32_t cols = net.weights.cols_array[k];
			for (uint32_t i = 0; i < cols; ++i) { // i == col_index
				double m = net.weights.data[net.weights.offset_positions[k] + thread_id * cols + i];
				double n = net.outputs.data[net.outputs.offset_positions[k-1] + i];
				accumulator += m * n;
			}
			uint32_t pos = net.outputs.offset_positions[k] + thread_id;
			net.outputs.data[pos] = accumulator;
		}
		__syncthreads();
		//vector_add (&net->outputs.data[i], &net->biases.data[i], &net->outputs.data[i]);
		if (thread_id < (net.outputs.size_array[k]) ) {
			uint32_t pos = net.outputs.offset_positions[k] + thread_id;
			net.outputs.data[pos] += net.biases.data[pos];
		}
		__syncthreads();
		//vector_copy (&net->outputs.data[i], &net->zs.data[i]); // vector_copy(src, dest)
		if (thread_id < (net.outputs.size_array[k]) ) {
			uint32_t pos = net.outputs.offset_positions[k] + thread_id;
			net.zs.data[pos] = net.outputs.data[pos];
		}
		__syncthreads();
		//vector_vectorise (&net->outputs.data[i], &sigmoid);
		if (thread_id < (net.outputs.size_array[k]) ) {
			uint32_t pos = net.outputs.offset_positions[k] + thread_id;
			net.outputs.data[pos] = device_sigmoid(net.outputs.data[pos]);
		}
		__syncthreads();
	}

	// step 3: network_get_output_error
	uint32_t output_layer_index = net.outputs.num_vectors - 1;

	// create expected output vector y from label
	double *expected_output_y = new double[net.outputs.size_array[output_layer_index]];
	for (uint32_t i = 0; i < net.outputs.size_array[output_layer_index]; i++) {
		expected_output_y[i] = 0.0;
	}
	expected_output_y[label] = 1.0;
	// create cost_derivative vector
	double *cost_deriv = new double[net.outputs.size_array[output_layer_index]];
	//vector_subtract(&net->outputs.data[output_layer_index], &expected_output_y, &cost_deriv)
	for (uint32_t i = 0; i < net.outputs.size_array[output_layer_index]; i++) {
		uint32_t pos = net.outputs.offset_positions[output_layer_index] + i;
		cost_deriv[i] = net.outputs.data[pos]  - expected_output_y[i];
	}
	//vector_copy (&net->zs.data[output_layer_index], &net->output_delta.data[output_layer_index]); // vector_copy(src, dest)
	if (thread_id < (net.output_delta.size_array[output_layer_index]) ) {
		uint32_t pos = net.output_delta.offset_positions[output_layer_index] + thread_id;
		net.output_delta.data[pos] = net.zs.data[pos];
	}
	__syncthreads();
	//vector_vectorise (&net->output_delta.data[output_layer_index],&sigmoid_prime);
	if (thread_id < (net.output_delta.size_array[output_layer_index]) ) {
		uint32_t pos = net.output_delta.offset_positions[output_layer_index] + thread_id;
		net.output_delta.data[pos] = device_sigmoid_prime(net.output_delta.data[pos]);
	}
	__syncthreads();

	//vector_product(&net->output_delta.data[output_layer_index], &cost_deriv, &net->output_delta.data[output_layer_index]);
	if (thread_id < (net.output_delta.size_array[output_layer_index]) ) {
		uint32_t pos = net.output_delta.offset_positions[output_layer_index] + thread_id;
		net.output_delta.data[pos] = net.output_delta.data[pos] * cost_deriv[thread_id];
	}
	__syncthreads();

	delete expected_output_y;
	delete cost_deriv;


	// step 4: network_accumulate_cfgs (net, output_layer_index);
	// vector_add (&net->output_delta.data[layer], &net->nabla_b.data[layer], &net->nabla_b.data[layer]);
	if (thread_id < (net.nabla_b.size_array[output_layer_index]) ) {
		uint32_t pos = net.nabla_b.offset_positions[output_layer_index] + thread_id;
		net.nabla_b.data[pos] += net.output_delta.data[pos];
	}
	__syncthreads();

	//col_vector_multiply_row_vector_with_sum(&net->output_delta.data[layer], &net->outputs.data[layer - 1], &net->nabla_w.data[layer]);
	uint32_t rows = net.nabla_w.rows_array[output_layer_index];
	uint32_t cols = net.nabla_w.cols_array[output_layer_index];
	if (thread_id < rows * cols) {
		uint32_t i_idx = thread_id / cols;
		uint32_t j_idx = thread_id % cols;
		uint32_t matrix_pos = net.nabla_w.offset_positions[output_layer_index] + thread_id ;
		uint32_t col_vector_pos = net.output_delta.offset_positions[output_layer_index] + i_idx;
		uint32_t row_vector_pos = net.outputs.offset_positions[output_layer_index - 1] + j_idx; // IMP: layer-1!
		net.nabla_w.data[matrix_pos] += net.output_delta.data[col_vector_pos] * net.outputs.data[row_vector_pos];
	}
	__syncthreads();

	// step 5: backpropagate
    for (int32_t l = output_layer_index - 1; l >= 0; --l)
    {
        //vector_copy (&net->zs.data[l], &net->output_delta.data[l]);
    	if (thread_id < (net.output_delta.size_array[l]) ) {
    		uint32_t pos = net.output_delta.offset_positions[l] + thread_id;
    		net.output_delta.data[pos] = net.zs.data[pos];
    	}
    	__syncthreads();

        //vector_vectorise (&net->output_delta.data[l], &sigmoid_prime);
    	if (thread_id < (net.output_delta.size_array[l]) ) {
    		uint32_t pos = net.output_delta.offset_positions[l] + thread_id;
    		net.output_delta.data[pos] = device_sigmoid_prime(net.output_delta.data[pos]);
    	}
    	__syncthreads();


        // create a copy of weights.data[l+1] matrix and then transpose
        uint32_t rows = net.weights.rows_array[l+1];
        uint32_t cols = net.weights.cols_array[l+1];
        //double* tmp_matrix = new double[rows * cols];
        //matrix_copy(&net->weights.data[l+1], &tmp_matrix); // matrix_copy(src, dest)
        //matrix_transpose(&tmp_matrix);

        if (thread_id < rows * cols) {
        	uint32_t i_idx = thread_id / cols;
        	uint32_t j_idx = thread_id % cols;
        	net.store2.data[j_idx * rows + i_idx] = net.weights.data[net.weights.offset_positions[l+1] + thread_id];
        }
        __syncthreads();

        // Y = alpha(A^T) + beta(Y)
        //matrix_vector_multiply (&tmp_matrix, &net->output_delta.data[l + 1], &tmp_vector);
		if (thread_id < cols) { // thread_id == row_index
			double accumulator = 0.0;
			for (uint32_t i = 0; i < rows; ++i) { // i == col_index
				double m = net.store2.data[thread_id * rows + i];
				double n = net.output_delta.data[net.output_delta.offset_positions[l+1] + i];
				accumulator += m * n;
			}
			net.store1.data[thread_id] = accumulator;
		}
		__syncthreads();


        // Back-propagated delta
        //vector_product (&tmp_vector, &net->output_delta.data[l], &net->output_delta.data[l]);
		if (thread_id < (net.output_delta.size_array[l]) ) {
			uint32_t pos = net.output_delta.offset_positions[l] + thread_id;
			net.output_delta.data[pos] = net.output_delta.data[pos] * net.store1.data[thread_id];
		}
		__syncthreads();

        //network_accumulate_cfgs (net, l);
		// vector_add (&net->output_delta.data[layer], &net->nabla_b.data[layer], &net->nabla_b.data[layer]);
		if (thread_id < (net.nabla_b.size_array[l]) ) {
			uint32_t pos = net.nabla_b.offset_positions[l] + thread_id;
			net.nabla_b.data[pos] += net.output_delta.data[pos];
		}
		__syncthreads();
		if ( l >= 1) {
			//col_vector_multiply_row_vector_with_sum(&net->output_delta.data[layer], &net->outputs.data[layer - 1], &net->nabla_w.data[layer]);
			rows = net.nabla_w.rows_array[l];
			cols = net.nabla_w.cols_array[l];
			if (thread_id < rows * cols) {
				uint32_t i_idx = thread_id / cols;
				uint32_t j_idx = thread_id % cols;
				uint32_t matrix_pos = net.nabla_w.offset_positions[l] + thread_id ;
				uint32_t col_vector_pos = net.output_delta.offset_positions[l] + i_idx;
				uint32_t row_vector_pos = net.outputs.offset_positions[l - 1] + j_idx; // IMP: layer-1!
				net.nabla_w.data[matrix_pos] += net.output_delta.data[col_vector_pos] * net.outputs.data[row_vector_pos];
			}
		} else {
			//col_vector_multiply_row_vector_with_sum(&net->output_delta.data[layer], &net->inputs, &net->nabla_w.data[layer]);
			rows = net.nabla_w.rows_array[l];
			cols = net.nabla_w.cols_array[l];
			uint32_t num_elements = rows * cols;
			uint32_t num_iter = 1;
			if (num_elements > num_threads) {
				num_iter = (num_elements % num_threads == 0) ? (num_elements / num_threads) : (num_elements / num_threads) + 1 ;
			}
			while (num_iter > 0) {
				uint32_t e_idx = (num_iter-1)*num_threads + thread_id;
				if (e_idx < num_elements) {
					uint32_t i_idx = e_idx / cols;
					uint32_t j_idx = e_idx % cols;
					uint32_t matrix_pos = net.nabla_w.offset_positions[l] + e_idx ;
					uint32_t col_vector_pos = net.output_delta.offset_positions[l] + i_idx;
					net.nabla_w.data[matrix_pos] += net.output_delta.data[col_vector_pos] * net.inputs.data[j_idx];
				}
				num_iter--;
			}

		}
		__syncthreads();


        //delete tmp_vector;
        //delete tmp_matrix;
    }

}

/*
 * backpropagation.cuh
 *
 *  Created on: 15-Dec-2015
 *      Author: saketsaurabh
 */

#ifndef BACKPROPAGATION_CUH_
#define BACKPROPAGATION_CUH_

__global__ void backpropagation_kernel(device_network_t net,
									const data_t data_d,
									const uint32_array_t rand_index,
									const uint32_t beginIndex,
									const uint32_t endIndex);


#endif /* BACKPROPAGATION_CUH_ */

/**
 * Filename: main.cpp
 * Authors: Saket Saurabh, Shashank Gupta
 * Language: C++
 * To Compile: Please check README.txt
 * Description: The main driver program that loads the data, initializes the network,
 * 				runs the stochastic gradient descent and reports the results.
 */

#include<cstdio>
#include<iostream>
#include<ctime>

#include "data_loader.h"
#include "data_types.h"

// constants
#define VALIDATION_DATA_CHUNK_SIZE 10000
#define MINI_BATCH_SIZE 10
#define EPOCHS 10
#define ETA 3.0
#define RANDOM_MEAN 0.0
#define RANDOM_STDDEV 1.0

// forward declarations
extern bool network_allocate (network_t * const network);
extern void network_random_init (network_t * const network, const double mean, const double stddev);
extern void network_sgd (network_t * const network,
             const data_t * const data,
             const data_t * const test_data);
extern void network_free (network_t * const network);


// Number of nodes in each layer of the network
uint32_t nodes[] = { 784, 30, 10 };

// path of the data files
const char * images_file = "../mnist/train-images-idx3-ubyte";
const char * labels_file = "../mnist/train-labels-idx1-ubyte";


int main(int argc, char** argv) {

	bool err;
	printf ("Loading images and labels...\n");
	data_t data;
	err = read_all_data (&data, images_file, labels_file);

	printf ("Setting up network...\n");
	network_t network;
	uint32_t layers = sizeof(nodes) / sizeof(nodes[0]);

    printf ("Node structure: ");
    for (uint32_t i = 0; i < layers - 1; ++i) {
        printf ("%i x ", nodes[i]);
    }
    printf ("%i.\n", nodes[layers - 1]);

    network.nodes.size = layers;
    network.nodes.data = nodes;
    network.epochs = EPOCHS;
    network.mini_batch_size = MINI_BATCH_SIZE;
    network.eta = ETA;

    err = network_allocate (&network);

    printf ("Initializing network...\n");
    network_random_init (&network, RANDOM_MEAN, RANDOM_STDDEV);

	// Split off a chunk of data for testing
	data_t test_data;
	partition_data(&data, &test_data, VALIDATION_DATA_CHUNK_SIZE);

	printf ("Stochastic gradient descent...\n");
	std::clock_t begin = clock();				// start the timer
	network_sgd (&network, &data, &test_data);
	std::clock_t end = clock();					// stop the timer
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("Time taken to train the network: %f seconds \n", elapsed_secs);

	// clean-up
	network_free (&network);
	images_free (&data.images);
	labels_free (&data.labels);

	return 0;

}

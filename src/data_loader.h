/*
 * data_loader.h
 *
 *  Created on: 13-Dec-2015
 *      Author: saketsaurabh
 */

#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include "data_types.h"

int32_t extract_header_line (const uint8_t * const buf);

bool read_all_data (data_t * const data,
               const char * images_file,
               const char * labels_file);

bool images_read_data (images_t * const image_data, const char * images_file);

bool images_allocate (images_t * const image_data, uint32_t pixels);

void images_free (images_t * const image_data);

void images_load_pixels (images_t * const image_data,
                    const uint32_t pixels,
                    FILE * const fp);

void images_print_stats (const images_t * const image_data);

bool labels_read_data (labels_t * const label_data, const char * labels_file);

bool labels_allocate (labels_t * const label_data);

void labels_free (labels_t * const label_data);

void labels_print_stats (const labels_t * const label_data);

void partition_data (data_t * const data,
                data_t * const test_data,
                const uint32_t chunk_size);





#endif /* DATA_LOADER_H_ */

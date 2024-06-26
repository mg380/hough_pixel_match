/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xf_houghlines_config.h"

#include <cstring>

extern "C" {

#define ACCUMULATOR_HEIGHT 1000
#define ACCUMULATOR_WIDTH 360
#define ACCUMULATOR_SLICE_WIDTH 2
#define ACCUMULATOR_DEPTH 360

static ap_fixed<16, 1, AP_RND> sinval_local[ACCUMULATOR_WIDTH] = {
	0.000000, 0.008727, 0.017452, 0.026177, 0.034899, 0.043619, 0.052336, 0.061049, 0.069756, 0.078459, 0.087156,
	0.095846, 0.104528, 0.113203, 0.121869, 0.130526, 0.139173, 0.147809, 0.156434, 0.165048, 0.173648, 0.182236,
	0.190809, 0.199368, 0.207912, 0.216440, 0.224951, 0.233445, 0.241922, 0.250380, 0.258819, 0.267238, 0.275637,
	0.284015, 0.292372, 0.300706, 0.309017, 0.317305, 0.325568, 0.333807, 0.342020, 0.350207, 0.358368, 0.366501,
	0.374607, 0.382684, 0.390731, 0.398749, 0.406737, 0.414693, 0.422618, 0.430511, 0.438371, 0.446198, 0.453991,
	0.461749, 0.469472, 0.477159, 0.484810, 0.492424, 0.499980, 0.507539, 0.515038, 0.522499, 0.529920, 0.537300,
	0.544639, 0.551937, 0.559193, 0.566407, 0.573577, 0.580703, 0.587786, 0.594823, 0.601815, 0.608762, 0.615662,
	0.622515, 0.629321, 0.636079, 0.642788, 0.649448, 0.656059, 0.662620, 0.669131, 0.675591, 0.681999, 0.688355,
	0.694659, 0.700910, 0.707107, 0.713251, 0.719340, 0.725375, 0.731354, 0.737278, 0.743145, 0.748956, 0.754710,
	0.760406, 0.766045, 0.771625, 0.777146, 0.782609, 0.788011, 0.793354, 0.798636, 0.803857, 0.809017, 0.814116,
	0.819152, 0.824127, 0.829038, 0.833886, 0.838671, 0.843392, 0.848048, 0.852641, 0.857168, 0.861629, 0.866026,
	0.870356, 0.874620, 0.878817, 0.882948, 0.887011, 0.891007, 0.894934, 0.898794, 0.902585, 0.906308, 0.909961,
	0.913545, 0.917060, 0.920505, 0.923879, 0.927184, 0.930417, 0.933580, 0.936672, 0.939692, 0.942641, 0.945518,
	0.948323, 0.951056, 0.953717, 0.956305, 0.958820, 0.961261, 0.963630, 0.965926, 0.968147, 0.970295, 0.972370,
	0.974370, 0.976296, 0.978147, 0.979924, 0.981627, 0.983255, 0.984807, 0.986285, 0.987688, 0.989016, 0.990268,
	0.991445, 0.992546, 0.993572, 0.994522, 0.995396, 0.996195, 0.996917, 0.997564, 0.998135, 0.998629, 0.999048,
	0.999391, 0.999657, 0.999848, 0.999962, 0.999980, 0.999962, 0.999848, 0.999657, 0.999391, 0.999048, 0.998629,
	0.998135, 0.997564, 0.996917, 0.996195, 0.995396, 0.994522, 0.993572, 0.992546, 0.991445, 0.990268, 0.989016,
	0.987688, 0.986285, 0.984807, 0.983255, 0.981627, 0.979924, 0.978147, 0.976296, 0.97437,  0.97237,  0.970295,
	0.968147, 0.965926, 0.96363,  0.961261, 0.95882,  0.956305, 0.953717, 0.951056, 0.948323, 0.945518, 0.942641,
	0.939692, 0.936672, 0.93358,  0.930417, 0.927184, 0.923879, 0.920505, 0.91706,  0.913545, 0.909961, 0.906308,
	0.902585, 0.898794, 0.894934, 0.891007, 0.887011, 0.882948, 0.878817, 0.87462,  0.870356, 0.866026, 0.861629,
	0.857168, 0.852641, 0.848048, 0.843392, 0.838671, 0.833886, 0.829038, 0.824127, 0.819152, 0.814116, 0.809017,
	0.803857, 0.798636, 0.793354, 0.788011, 0.782609, 0.777146, 0.771625, 0.766045, 0.760406, 0.75471,  0.748956,
	0.743145, 0.737278, 0.731354, 0.725375, 0.71934,  0.713251, 0.707107, 0.70091,  0.694659, 0.688355, 0.681999,
	0.675591, 0.669131, 0.66262,  0.656059, 0.649448, 0.642788, 0.636079, 0.629321, 0.622515, 0.615662, 0.608762,
	0.601815, 0.594823, 0.587786, 0.580703, 0.573577, 0.566407, 0.559193, 0.551937, 0.544639, 0.5373,   0.52992,
	0.522499, 0.515038, 0.507539, 0.5,	  0.492424, 0.48481,  0.477159, 0.469472, 0.461749, 0.453991, 0.446198,
	0.438371, 0.430511, 0.422618, 0.414693, 0.406737, 0.398749, 0.390731, 0.382684, 0.374607, 0.366501, 0.358368,
	0.350207, 0.34202,  0.333807, 0.325568, 0.317305, 0.309017, 0.300706, 0.292372, 0.284015, 0.275637, 0.267238,
	0.258819, 0.25038,  0.241922, 0.233445, 0.224951, 0.21644,  0.207912, 0.199368, 0.190809, 0.182236, 0.173648,
	0.165048, 0.156434, 0.147809, 0.139173, 0.130526, 0.121869, 0.113203, 0.104528, 0.095846, 0.087156, 0.078459,
	0.069756, 0.061049, 0.052336, 0.043619, 0.034899, 0.026177, 0.017452, 0.008727};

static ap_fixed<16, 1, AP_RND> cosval_local[ACCUMULATOR_WIDTH] = {
	0.999980,  0.999962,  0.999848,  0.999657,  0.999391,  0.999048,  0.998629,  0.998135,  0.997564,  0.996917,
	0.996195,  0.995396,  0.994522,  0.993572,  0.992546,  0.991445,  0.990268,  0.989016,  0.987688,  0.986285,
	0.984807,  0.983255,  0.981627,  0.979924,  0.978147,  0.976296,  0.97437,   0.97237,   0.970295,  0.968147,
	0.965926,  0.96363,   0.961261,  0.95882,   0.956305,  0.953717,  0.951056,  0.948323,  0.945518,  0.942641,
	0.939692,  0.936672,  0.93358,   0.930417,  0.927184,  0.923879,  0.920505,  0.91706,   0.913545,  0.909961,
	0.906308,  0.902585,  0.898794,  0.894934,  0.891007,  0.887011,  0.882948,  0.878817,  0.87462,   0.870356,
	0.866026,  0.861629,  0.857168,  0.852641,  0.848048,  0.843392,  0.838671,  0.833886,  0.829038,  0.824127,
	0.819152,  0.814116,  0.809017,  0.803857,  0.798636,  0.793354,  0.788011,  0.782609,  0.777146,  0.771625,
	0.766045,  0.760406,  0.75471,   0.748956,  0.743145,  0.737278,  0.731354,  0.725375,  0.71934,   0.713251,
	0.707107,  0.70091,   0.694659,  0.688355,  0.681999,  0.675591,  0.669131,  0.66262,   0.656059,  0.649448,
	0.642788,  0.636079,  0.629321,  0.622515,  0.615662,  0.608762,  0.601815,  0.594823,  0.587786,  0.580703,
	0.573577,  0.566407,  0.559193,  0.551937,  0.544639,  0.5373,	0.52992,   0.522499,  0.515038,  0.507539,
	0.5,	   0.492424,  0.48481,   0.477159,  0.469472,  0.461749,  0.453991,  0.446198,  0.438371,  0.430511,
	0.422618,  0.414693,  0.406737,  0.398749,  0.390731,  0.382684,  0.374607,  0.366501,  0.358368,  0.350207,
	0.34202,   0.333807,  0.325568,  0.317305,  0.309017,  0.300706,  0.292372,  0.284015,  0.275637,  0.267238,
	0.258819,  0.25038,   0.241922,  0.233445,  0.224951,  0.21644,   0.207912,  0.199368,  0.190809,  0.182236,
	0.173648,  0.165048,  0.156434,  0.147809,  0.139173,  0.130526,  0.121869,  0.113203,  0.104528,  0.095846,
	0.087156,  0.078459,  0.069756,  0.061049,  0.052336,  0.043619,  0.034899,  0.026177,  0.017452,  0.008727,
	0.000000,  -0.008727, -0.017452, -0.026177, -0.034899, -0.043619, -0.052336, -0.061049, -0.069756, -0.078459,
	-0.087156, -0.095846, -0.104528, -0.113203, -0.121869, -0.130526, -0.139173, -0.147809, -0.156434, -0.165048,
	-0.173648, -0.182236, -0.190809, -0.199368, -0.207912, -0.216440, -0.224951, -0.233445, -0.241922, -0.250380,
	-0.258819, -0.267238, -0.275637, -0.284015, -0.292372, -0.300706, -0.309017, -0.317305, -0.325568, -0.333807,
	-0.342020, -0.350207, -0.358368, -0.366501, -0.374607, -0.382684, -0.390731, -0.398749, -0.406737, -0.414693,
	-0.422618, -0.430511, -0.438371, -0.446198, -0.453991, -0.461749, -0.469472, -0.477159, -0.484810, -0.492424,
	-0.499980, -0.507539, -0.515038, -0.522499, -0.529920, -0.537300, -0.544639, -0.551937, -0.559193, -0.566407,
	-0.573577, -0.580703, -0.587786, -0.594823, -0.601815, -0.608762, -0.615662, -0.622515, -0.629321, -0.636079,
	-0.642788, -0.649448, -0.656059, -0.662620, -0.669131, -0.675591, -0.681999, -0.688355, -0.694659, -0.700910,
	-0.707107, -0.713251, -0.719340, -0.725375, -0.731354, -0.737278, -0.743145, -0.748956, -0.754710, -0.760406,
	-0.766045, -0.771625, -0.777146, -0.782609, -0.788011, -0.793354, -0.798636, -0.803857, -0.809017, -0.814116,
	-0.819152, -0.824127, -0.829038, -0.833886, -0.838671, -0.843392, -0.848048, -0.852641, -0.857168, -0.861629,
	-0.866026, -0.870356, -0.874620, -0.878817, -0.882948, -0.887011, -0.891007, -0.894934, -0.898794, -0.902585,
	-0.906308, -0.909961, -0.913545, -0.917060, -0.920505, -0.923879, -0.927184, -0.930417, -0.933580, -0.936672,
	-0.939692, -0.942641, -0.945518, -0.948323, -0.951056, -0.953717, -0.956305, -0.958820, -0.961261, -0.963630,
	-0.965926, -0.968147, -0.970295, -0.972370, -0.974370, -0.976296, -0.978147, -0.979924, -0.981627, -0.983255,
	-0.984807, -0.986285, -0.987688, -0.989016, -0.990268, -0.991445, -0.992546, -0.993572, -0.994522, -0.995396,
	-0.996195, -0.996917, -0.997564, -0.998135, -0.998629, -0.999048, -0.999391, -0.999657, -0.999848, -0.999962};

void houghlines_accel(
	ap_uint<PTR_WIDTH>* img_in_global, short threshold, short maxlines, int *rows_out, int *cols_out, int *votes_out, int rows, int cols) {
	printf("Kernel started\n");

	size_t image_in_size_bytes = rows * cols * sizeof(unsigned char);
	ap_uint<PTR_WIDTH> img_in_local[image_in_size_bytes];

	// Move image into a high speed memory on the FPGA fabric
	std::memcpy(img_in_local, img_in_global, image_in_size_bytes);

	int accumulator_votes[ACCUMULATOR_SLICE_WIDTH][ACCUMULATOR_HEIGHT];
	int accumulator_pixels[ACCUMULATOR_SLICE_WIDTH][ACCUMULATOR_HEIGHT][ACCUMULATOR_DEPTH][2];

	// clang-format off
	#pragma HLS INTERFACE m_axi		port=img_in_local	offset=slave  bundle=gmem0
	#pragma HLS INTERFACE s_axilite port=img_in_local
	#pragma HLS INTERFACE m_axi	 	port=rows_out  		offset=slave  bundle=gmem1
	#pragma HLS INTERFACE s_axilite port=rows_out
	#pragma HLS INTERFACE m_axi	 	port=cols_out  		offset=slave  bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=cols_out
	#pragma HLS INTERFACE m_axi	 	port=votes_out 		offset=slave  bundle=gmem3
	#pragma HLS INTERFACE s_axilite port=votes_out
	#pragma HLS INTERFACE s_axilite port=threshold
	#pragma HLS INTERFACE s_axilite port=maxlines
	#pragma HLS INTERFACE s_axilite port=rows
	#pragma HLS INTERFACE s_axilite port=cols	 
	#pragma HLS INTERFACE s_axilite port=return
	// clang-format on

	float radius_max = 2 * float(rows + cols);
	float conversion_factor = ACCUMULATOR_HEIGHT / float(radius_max);

	// Keep track of how many elements have been written to our output array
	int output_counter = 0;

	// Scan through slices in theta
	for (int theta_lower_index = 0; theta_lower_index < ACCUMULATOR_WIDTH; theta_lower_index += ACCUMULATOR_SLICE_WIDTH)
	{
		// Reset accumulator
		for (int theta_index = 0; theta_index < ACCUMULATOR_SLICE_WIDTH; theta_index++)
			for (int radius_index = 0; radius_index < ACCUMULATOR_HEIGHT; radius_index++)
				accumulator_votes[theta_index][radius_index] = 0;

		for (int row = 0; row < rows; row++)
			for (int col = 0; col < cols; col++)
			{
				int offset = (row * cols) + col;

				// Ensure only active pixel are included in the voting process
				if (img_in_local[offset] == 0)
					continue;

				for (int theta_index = 0; theta_index < ACCUMULATOR_SLICE_WIDTH; theta_index++)
				{
					int trig_index = theta_lower_index + theta_index;
					double radius = (col * cosval_local[trig_index]) + (row * sinval_local[trig_index]);
					int radius_index = int((radius + rows + cols) * conversion_factor);
					int votes = accumulator_votes[theta_index][radius_index];

					//printf("Theta Loop: %i\t%i\t%i\t%f\t%f\n", theta_index, radius_index, votes, conversion_factor, radius);

					if (votes >= ACCUMULATOR_DEPTH)
						continue;

					accumulator_pixels[theta_index][radius_index][votes][0] = row;
					accumulator_pixels[theta_index][radius_index][votes][1] = col;
					accumulator_votes[theta_index][radius_index]++;
				}
			}

		for (int theta_index = 0; theta_index < ACCUMULATOR_SLICE_WIDTH; theta_index++)
			for (int radius_index = 0; radius_index < ACCUMULATOR_HEIGHT; radius_index++)
			{
				int votes = accumulator_votes[theta_index][radius_index];

				//printf("Radius Loop: %i\t%i\t%i\n", theta_lower_index, radius_index, votes);

				if (votes > threshold)
				{
					for (int vote = 0; vote < votes; vote++)
					{
						int *pixel = accumulator_pixels[theta_index][radius_index][vote];

						if (output_counter >= (maxlines - 1))
							break;

						rows_out[output_counter] = pixel[0];
						cols_out[output_counter] = pixel[1];
						votes_out[output_counter] = votes;

						output_counter++;
					}

					if (output_counter >= maxlines)
						break;

					rows_out[output_counter] = -1;
					cols_out[output_counter] = -1;
					votes_out[output_counter] = -1;

					output_counter++;
				}
			}
	}

	//printf("Kernel finished\n");

	return;
} // End of kernel
} // End of extern C


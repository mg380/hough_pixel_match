/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <time.h>

#include "common/xf_headers.hpp"
#include "xcl2.hpp"
#include "xf_houghlines_config.h"

float sinvalt[360] = {
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
    0.522499, 0.515038, 0.507539, 0.5,      0.492424, 0.48481,  0.477159, 0.469472, 0.461749, 0.453991, 0.446198,
    0.438371, 0.430511, 0.422618, 0.414693, 0.406737, 0.398749, 0.390731, 0.382684, 0.374607, 0.366501, 0.358368,
    0.350207, 0.34202,  0.333807, 0.325568, 0.317305, 0.309017, 0.300706, 0.292372, 0.284015, 0.275637, 0.267238,
    0.258819, 0.25038,  0.241922, 0.233445, 0.224951, 0.21644,  0.207912, 0.199368, 0.190809, 0.182236, 0.173648,
    0.165048, 0.156434, 0.147809, 0.139173, 0.130526, 0.121869, 0.113203, 0.104528, 0.095846, 0.087156, 0.078459,
    0.069756, 0.061049, 0.052336, 0.043619, 0.034899, 0.026177, 0.017452, 0.008727};
float cosvalt[360] = {
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
    0.573577,  0.566407,  0.559193,  0.551937,  0.544639,  0.5373,    0.52992,   0.522499,  0.515038,  0.507539,
    0.5,       0.492424,  0.48481,   0.477159,  0.469472,  0.461749,  0.453991,  0.446198,  0.438371,  0.430511,
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

struct LinePolar {
    float rho;
    float angle;
};

struct hough_cmp_gt {
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const { return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2); }
    const int* aux;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat out_img;
    cv::Mat in_gray, in_gray1, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Create memory for output images:
    out_img.create(in_gray.rows, in_gray.cols, in_gray.depth());

    cv::Mat dst, cdst;
    cv::Canny(in_gray, dst, 50, 200, 3);

    cvtColor(dst, cdst, cv::COLOR_GRAY2BGR);

    std::vector<int> rows_out(LINESMAX);
    std::vector<int> cols_out(LINESMAX);
    std::vector<int> votes_out(LINESMAX);

    short threshold = 75;
    short maxlines = LINESMAX;

    int height = in_gray.rows;
    int width = in_gray.cols;

    cv::imwrite("/home/centos/outref.png", cdst);

    // OpenCL section:
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = LINESMAX * sizeof(int);

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_houghlines");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "houghlines_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_rows_out(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_cols_out(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_votes_out(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));


    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, threshold));
    OCL_CHECK(err, err = kernel.setArg(2, maxlines));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_rows_out));
    OCL_CHECK(err, err = kernel.setArg(4, buffer_cols_out));
    OCL_CHECK(err, err = kernel.setArg(5, buffer_votes_out));
    OCL_CHECK(err, err = kernel.setArg(6, height));
    OCL_CHECK(err, err = kernel.setArg(7, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       dst.data,            // Pointer to the data to copy
                                       nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_rows_out, // This buffers data will be read
                            CL_TRUE,          // blocking call
                            0,                // offset
                            image_out_size_bytes,
                            rows_out.data(), // Data will be stored here
                            nullptr, &event);

    queue.enqueueReadBuffer(buffer_cols_out, // This buffers data will be read
                            CL_TRUE,          // blocking call
                            0,                // offset
                            image_out_size_bytes,
                            cols_out.data(), // Data will be stored here
                            nullptr, &event);

    queue.enqueueReadBuffer(buffer_votes_out, // This buffers data will be read
							CL_TRUE,          // blocking call
							0,                // offset
							image_out_size_bytes,
							votes_out.data(), // Data will be stored here
							nullptr, &event);

    // Clean up:
    queue.finish();

    int line_colour = 0;

    printf("Started Drawing\n");

	for (int line_index = 0; (line_index < maxlines); line_index++)
	{
		int col = rows_out[line_index];
		int row = cols_out[line_index];
		int votes = votes_out[line_index];
		printf("Line Index:\t%i\t%i\t%i\t%i\n", line_index, row, col, votes);

		if ((row < 0) || (col < 0))
		{
			printf("New Line: %i\n", line_colour);
			line_colour++;
			continue;
		}

		switch (line_colour % 3)
		{
		case 0:
			cdst.at<cv::Vec3b>(col, row)[0] = 255;
			cdst.at<cv::Vec3b>(col, row)[1] = 0;
			cdst.at<cv::Vec3b>(col, row)[2] = 0;
			break;
		case 1:
			cdst.at<cv::Vec3b>(col, row)[0] = 0;
			cdst.at<cv::Vec3b>(col, row)[1] = 255;
			cdst.at<cv::Vec3b>(col, row)[2] = 0;
			break;
		case 2:
			cdst.at<cv::Vec3b>(col, row)[0] = 0;
			cdst.at<cv::Vec3b>(col, row)[1] = 0;
			cdst.at<cv::Vec3b>(col, row)[2] = 255;
			break;
		}
	}

	printf("Writing Image\n");
	cv::imwrite("/home/centos/outhls.png", cdst);
	printf("Wrote Images\n");

	return 0;
}

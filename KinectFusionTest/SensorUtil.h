#pragma once

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

#include "cuda_SimpleMatrixUtil.h"

namespace util
{
	namespace cuda
	{
		void bilateral_filter(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
		void subtract_foreground(float* d_output, float* d_input, float foreground_depth_min, float foreground_depth_max, unsigned int width, unsigned int height);
		void convert_depth2vertex(float4* d_output, float* d_input, float4x4 inv_intrinsics, unsigned int width, unsigned int height);
		void compute_normals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
	}
}
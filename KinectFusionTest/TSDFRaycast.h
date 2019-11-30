#pragma once

#include "cuda_SimpleMatrixUtil.h"
#include "opencv2/opencv.hpp"

namespace tsdf {
	namespace cuda {
		void depth_raycast(float4x4 intrinsic, 
						   float4x4 extrinsic,
						   uint     width, 
						   uint     height,
						   float3   voxel_grid_origin, 
						   uint3    voxel_grid_dim, 
						   float    voxel_size, 
						   float    trunc_margin,
						   float*   tsdf, 
						   float*   weight, 
						   float*   out_depth);
	}
}
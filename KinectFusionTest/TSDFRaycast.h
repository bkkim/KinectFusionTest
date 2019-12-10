#pragma once

#include "cuda_SimpleMatrixUtil.h"
#include "opencv2/opencv.hpp"

namespace tsdf {
	namespace cuda {
		void raycast(float4x4 intrinsic, 
			         float4x4 extrinsic,
			         uint     width, 
			         uint     height,
			         float3   volume_origin, 
			         uint3    volume_dim, 
			         float    voxel_size, 
			         float    trunc_margin,
			         float*   tsdf, 
			         float*   weight, 
			         float*   raycast_depth,
			         float4*  raycast_vertex,
			         float4*  raycast_normal,
			         uchar4*  raycast_color);
	}
}
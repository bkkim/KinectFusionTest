#pragma once

#include "cuda_SimpleMatrixUtil.h"
typedef unsigned char uchar;

namespace tsdf
{
	namespace cuda 
	{
		void update(float4x4 intrinsic,
					float4x4 extrinsic,
					uint     width,
					uint     height,
					float    *img_depth,
					uchar3   *img_color,
					uint3    voxel_dim,
					float3   voxel_origin,
					float    voxel_size,
					float    trunc_margin,
					float    *tsdf,
					float    *tsdf_weight,
					uchar3   *color,
					uchar    *color_weight);
	}
}
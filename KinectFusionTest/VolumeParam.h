#pragma once

#include "cuda_SimpleMatrixUtil.h"

struct VolumeParam
{
	float3 voxel_origin;
	uint3  voxel_dim;
	float  voxel_size;
	float  trunc_margin;
};
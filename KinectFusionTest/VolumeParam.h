#pragma once

#include "cuda_SimpleMatrixUtil.h"

struct VolumeParam
{
	float3 volume_origin;
	uint3  volume_dim;
	float  voxel_size;
	float  trunc_margin;
};
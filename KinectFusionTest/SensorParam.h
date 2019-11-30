#pragma once

#include "cuda_SimpleMatrixUtil.h"

struct SensorParam
{
public:
	SensorParam() {}

	uint		width;
	uint		height;
	float		depthMin;
	float		depthMax;
	float4x4	intrinsic;
	float4x4	extrinsic;
};
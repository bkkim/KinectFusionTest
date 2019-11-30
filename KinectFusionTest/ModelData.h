#pragma once

#include "SensorParam.h"

typedef unsigned char uchar;

class ModelData
{
public:
	ModelData(SensorParam& param) {
		create(param);
	}

	~ModelData() {
		destroy();
	}

private:
	void create(SensorParam& param)
	{
		uint size = param.width * param.height;

		cudaMalloc(&d_raycast_depth,  sizeof(float )*size);
		cudaMalloc(&d_raycast_vertex, sizeof(float4)*size);
		cudaMalloc(&d_raycast_normal, sizeof(float4)*size);
	}

	void destroy()
	{
		cudaFree(d_raycast_depth);
		cudaFree(d_raycast_vertex);
		cudaFree(d_raycast_normal);
	}

public:
	// Model data
	float  *d_raycast_depth;
	float4 *d_raycast_vertex;
	float4 *d_raycast_normal;
};
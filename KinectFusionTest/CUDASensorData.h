#pragma once

#include <helper_cuda.h>
#include "SensorData.h"

class CUDASensorData
{
public:

	CUDASensorData(SensorParam& param)
		: d_depth           (NULL)
		, d_color           (NULL)
		, d_depth_filtered  (NULL)
		, d_depth_foreground(NULL)
		, d_vertex          (NULL)
		, d_normal          (NULL)
	{
		create(param);
	}

	~CUDASensorData()
	{
		destroy();
	}

private:

	void create(SensorParam& param)
	{
		cudaMalloc(&d_depth,            sizeof(float )*param.width*param.height);
		cudaMalloc(&d_color,            sizeof(uchar3)*param.width*param.height);
		cudaMalloc(&d_depth_filtered,   sizeof(float )*param.width*param.height);
		cudaMalloc(&d_depth_foreground, sizeof(float )*param.width*param.height);
		cudaMalloc(&d_vertex,           sizeof(float4)*param.width*param.height);
		cudaMalloc(&d_normal,           sizeof(float4)*param.width*param.height);
	}

	void destroy()
	{
		cudaFree(d_depth);
		cudaFree(d_color);
		cudaFree(d_depth_filtered);
		cudaFree(d_depth_foreground);
		cudaFree(d_vertex);
		cudaFree(d_normal);
	}

public:
	// Derived data
	float		*d_depth;
	uchar3		*d_color;
	float		*d_depth_filtered;
	float		*d_depth_foreground;
	float4		*d_vertex;
	float4		*d_normal;

};

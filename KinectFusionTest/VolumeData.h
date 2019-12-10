#pragma once

#include "VolumeParam.h"

typedef unsigned char uchar;

//////////////////////////////////////////////////////////////////////////
//!
extern "C" void launch_reset(uint3 voxel_grid, float* tsdf, float* tsdf_weight, uchar3* color, uchar* color_weight);


class VolumeData
{
public:
	VolumeData(VolumeParam& param) {
		create(param);
	}

	~VolumeData() {
		destroy();
	}

private:
	void create(VolumeParam& param) 
	{
		uint volume_size = param.volume_dim.x * param.volume_dim.y * param.volume_dim.z;
		
		cudaMalloc(&d_tsdf,         sizeof(float )*volume_size);
		cudaMalloc(&d_tsdf_weight,  sizeof(float )*volume_size);
		cudaMalloc(&d_color,        sizeof(uchar3)*volume_size);
		cudaMalloc(&d_color_weight, sizeof(uchar )*volume_size);
	}

	void destroy()
	{
		cudaFree(d_tsdf);
		cudaFree(d_tsdf_weight);
		cudaFree(d_color);
		cudaFree(d_color_weight);
	}

public:
	void reset(VolumeParam& param)
	{
		launch_reset(param.volume_dim, d_tsdf, d_tsdf_weight, d_color, d_color_weight);
	}

public:
	// TSDF Volume data
	float	*d_tsdf;
	float	*d_tsdf_weight;
	uchar3	*d_color;
	uchar	*d_color_weight; // Alpha
};
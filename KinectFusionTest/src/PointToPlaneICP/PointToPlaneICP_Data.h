#pragma once


#include "PointToPlaneICP_Params.h"

class PointToPlaneICP_Data
{
public:
	PointToPlaneICP_Data()
		: m_level(1)
		, d_frame_vertex(NULL)
		, d_frame_normal(NULL)
		, d_model_vertex(NULL)
		, d_model_normal(NULL)
		, d_corre_vertex(NULL)
		, d_corre_normal(NULL)
		, d_A(NULL)
		, d_b(NULL)
	{}

	void allocation(const PointToPlaneICP_Params& params)
	{
		m_level = params.level;
		size_t size = params.width*params.height;
		checkCudaErrors(cudaMalloc(&d_frame_vertex, sizeof(float4)* size));
		checkCudaErrors(cudaMalloc(&d_frame_normal, sizeof(float4)* size));
		checkCudaErrors(cudaMalloc(&d_model_vertex, sizeof(float4)* size));
		checkCudaErrors(cudaMalloc(&d_model_normal, sizeof(float4)* size));
		checkCudaErrors(cudaMalloc(&d_corre_vertex, sizeof(float4)* size));
		checkCudaErrors(cudaMalloc(&d_corre_normal, sizeof(float4)* size));
		checkCudaErrors(cudaMalloc(&d_A,            sizeof(float)*  size * 6));
		checkCudaErrors(cudaMalloc(&d_b,            sizeof(float)*  size));
	}

	void free()
	{
		checkCudaErrors(cudaFree(d_frame_vertex));
		checkCudaErrors(cudaFree(d_frame_normal));
		checkCudaErrors(cudaFree(d_model_vertex));
		checkCudaErrors(cudaFree(d_model_normal));
		checkCudaErrors(cudaFree(d_corre_vertex));
		checkCudaErrors(cudaFree(d_corre_normal));
		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_b));
		
		d_frame_vertex = NULL;
		d_frame_normal = NULL;
		d_model_vertex = NULL;
		d_model_normal = NULL;
		d_corre_vertex = NULL;
		d_corre_normal = NULL;
		d_A = NULL;
		d_b = NULL;
	}

public:
	int     m_level = 1;
	float4* d_frame_vertex = NULL;
	float4* d_frame_normal = NULL;
	float4* d_model_vertex = NULL;
	float4* d_model_normal = NULL;
	float4* d_corre_vertex = NULL;	// for correspondences 
	float4* d_corre_normal = NULL;  // for correspondences, its type is float4* for using w value. 
	float*  d_A = NULL;	            // for system
	float*  d_b = NULL;	            // for system
};
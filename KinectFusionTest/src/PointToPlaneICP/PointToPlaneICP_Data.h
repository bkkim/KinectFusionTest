#pragma once


#include "PointToPlaneICP_Params.h"

class PointToPlaneICP_Data
{
public:
	PointToPlaneICP_Data()
		: d_corres_vertex(NULL)
		, d_corres_normal(NULL)
		, d_A(NULL)
		, d_b(NULL)
	{}

	void allocation(const PointToPlaneICP_Params& params)
	{
		checkCudaErrors(cudaMalloc(&d_corres_vertex, sizeof(float4)* params.width*params.height));
		checkCudaErrors(cudaMalloc(&d_corres_normal, sizeof(float4)* params.width*params.height));
		checkCudaErrors(cudaMalloc(&d_A, sizeof(float)* params.width*params.height * 6));
		checkCudaErrors(cudaMalloc(&d_b, sizeof(float)* params.width*params.height));
	}

	void free()
	{
		checkCudaErrors(cudaFree(d_corres_vertex));
		checkCudaErrors(cudaFree(d_corres_normal));
		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_b));
		
		d_corres_vertex = NULL;
		d_corres_normal = NULL;
		d_A = NULL;
		d_b = NULL;
	}

public:
	float4* d_corres_vertex = NULL;	// for correspondences 
	float4* d_corres_normal = NULL; // for correspondences, its type is float4* for using w value. 
	float*  d_A = NULL;				// for system
	float*  d_b = NULL;				// for system
};
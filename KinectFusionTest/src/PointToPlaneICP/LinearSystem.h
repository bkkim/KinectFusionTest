#pragma once

// Linear System Build on the GPU for ICP

#include "Eigen.h"
#include "cuda_SimpleMatrixUtil.h"

#include "LinearSystemConfidence.h"

class LinearSystem
{
public:
	LinearSystem(unsigned int width, unsigned int height);
	~LinearSystem();

	void apply(float4*                 src_vertex,
			   float4*                 dst_vertex,
			   float4*                 dst_normal,
			   float4x4&               deltaTransform,
			   unsigned int            width,
			   unsigned int            height,
			   Matrix6x7f&             res,
			   LinearSystemConfidence& confidence);

	//! builds AtA, AtB, and confidences
	Matrix6x7f reductionSystem(const float* data, unsigned int nElems, LinearSystemConfidence& confidence);

private:

	float* d_output;
	float* h_output;
};

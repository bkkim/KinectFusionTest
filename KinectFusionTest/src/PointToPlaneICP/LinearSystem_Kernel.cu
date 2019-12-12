
//#include <helper_cuda.h>
//#include <helper_math.h>

#include "cuda_SimpleMatrixUtil.h"

/////////////////////////////////////////////////////
// Defines
/////////////////////////////////////////////////////
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 30
#endif

/////////////////////////////////////////////////////
// Shared Memory
/////////////////////////////////////////////////////
__shared__ float bucket2[ARRAY_SIZE*BLOCK_SIZE];

/////////////////////////////////////////////////////
// Helper Functions
/////////////////////////////////////////////////////
__device__ inline void addToLocalScanElement(uint inpGTid, uint resGTid, volatile float* shared)
{
	//	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		shared[ARRAY_SIZE*resGTid + i] += shared[ARRAY_SIZE*inpGTid + i];
	}
}

__device__ inline void CopyToResultScanElement(uint GID, float* output)
{
	//	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID + i] = bucket2[i];
	}
}

__device__ inline void SetZeroScanElement(uint GTid)
{
	//	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		bucket2[GTid*ARRAY_SIZE + i] = 0.0f;
	}
}

/////////////////////////////////////////////////////
// Linearized System Matrix
/////////////////////////////////////////////////////

// Matrix Struct
struct Float1x6
{
	float data[6];
};

// Arguments: q moving point, n normal target
__device__ inline  Float1x6 buildRowSystemMatrix(float3 s, float3 n, float w)
{
	Float1x6 row;
	row.data[0] = n.z*s.y - n.y*s.z;
	row.data[1] = n.x*s.z - n.z*s.x;
	row.data[2] = n.y*s.x - n.x*s.y;

	row.data[3] = n.x;
	row.data[4] = n.y;
	row.data[5] = n.z;

	return row;
}

// Arguments: d target point, s moving point, n normal target
__device__ inline  float buildRowRHS(float3 s, float3 d, float3 n, float w)
{
	return n.x*(d.x - s.x) + n.y*(d.y - s.y) + n.z*(d.z - s.z);
}

// Arguments: d target point, s moving point, n normal target
__device__ inline  void addToLocalSystem(float3 s, float3 d, float3 n, float weight, uint GTid)
{
	const Float1x6 Ai = buildRowSystemMatrix(s, n, weight);
	const float    bi = buildRowRHS(s, d, n, weight);
	uint linRowStart = 0;

	for (uint i = 0; i<6; i++) 
	{
		for (uint j = i; j<6; j++) 
		{
			bucket2[ARRAY_SIZE*GTid + linRowStart + j - i] += weight*Ai.data[i] * Ai.data[j];
		}

		linRowStart += 6 - i;

		bucket2[ARRAY_SIZE*GTid + 21 + i] += weight*Ai.data[i] * bi;
	}

	const float dN = dot(s - d, n);
	bucket2[ARRAY_SIZE*GTid + 27] += weight*dN*dN;		//residual
	bucket2[ARRAY_SIZE*GTid + 28] += weight;			//corr weight
	bucket2[ARRAY_SIZE*GTid + 29] += 1.0f;				//corr number
}

/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

__device__ inline void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid, bucket2);
	addToLocalScanElement(GTid + 16, GTid, bucket2);
	addToLocalScanElement(GTid + 8, GTid, bucket2);
	addToLocalScanElement(GTid + 4, GTid, bucket2);
	addToLocalScanElement(GTid + 2, GTid, bucket2);
	addToLocalScanElement(GTid + 1, GTid, bucket2);
}

__global__ void scanScanElementsCS(unsigned int width,
								   unsigned int height,
								   float* output,
								   float4* input,
								   float4* target,
								   float4* targetNormals,
								   float4x4 deltaTransform,
								   unsigned int localWindowSize)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;

	// Set system to zero
	SetZeroScanElement(threadIdx.x);

	//Locally sum small window
#pragma unroll
	for (uint i = 0; i < localWindowSize; i++)
	{
		const int index1D = localWindowSize * x + i;
		const uint2 index = make_uint2(index1D % width, index1D / width);

		if (index.x < width && index.y < height)
		{
			if (target[index1D].x != MINF && input[index1D].x != MINF && targetNormals[index1D].x != MINF) 
			{
				// Referred to Kok-Lim Low's paper.
				const float3 s = deltaTransform * make_float3(input[index1D]);
				const float3 d = make_float3(target[index1D]);
				const float3 n = make_float3(targetNormals[index1D]);
				const float  w = targetNormals[index1D].w; // weight

				// Compute Linearized System
				addToLocalSystem(s, d, n, w, threadIdx.x);
			}
		}
	}
	__syncthreads();

	if (threadIdx.x < (BLOCK_SIZE/2))
		warpReduce(threadIdx.x);

	// Copy to output texture
	if (threadIdx.x == 0)
		CopyToResultScanElement(blockIdx.x, output);
}

extern "C" void
launch_build_linear_system(unsigned int width,
						   unsigned int height,
						   float*       output,
						   float4*      src_vertex,
						   float4*      dst_vertex,
						   float4*      dst_normal,
						   float4x4     deltaTransform,
						   unsigned int localWindowSize,
						   unsigned int blockSizeInt)
{
	const unsigned int numElements = width * height;

	dim3 blockSize(blockSizeInt, 1, 1);
	dim3 gridSize((numElements + blockSizeInt*localWindowSize - 1) / (blockSizeInt*localWindowSize), 1, 1);

	scanScanElementsCS <<< gridSize, blockSize >>> (width, 
													height, 
													output, 
													src_vertex, 
													dst_vertex, 
													dst_normal, 
													deltaTransform, 
													localWindowSize);

#ifdef _DEBUG
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError(__FUNCTION__);
#endif
}
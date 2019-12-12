#include "LinearSystem.h"

//////////////////////////////////////////////////////////////////////////
//!
extern "C" void
launch_build_linear_system(unsigned int _width,
						   unsigned int _height,
						   float*       _output,
						   float4*      _src_vertex,
						   float4*      _dst_vertex,
						   float4*      _dst_normal,
						   float4x4     _deltaTransform,
						   unsigned int _localWindowSize,
						   unsigned int _blockSizeInt);
//////////////////////////////////////////////////////////////////////////


LinearSystem::LinearSystem(unsigned int width, unsigned int height)
{
	const unsigned int localWindowSize = 24;
	const unsigned int blockSize = 64;
	const unsigned int dimX = (unsigned int)ceil(((float)width*height) / (localWindowSize*blockSize));

	checkCudaErrors(cudaMalloc(&d_output, 30 * sizeof(float)*dimX));
	h_output = new float[30 * dimX];
}

LinearSystem::~LinearSystem()
{
	if (d_output) {
		checkCudaErrors(cudaFree(d_output));
	}

	if (h_output)
		delete[] h_output;
}

void LinearSystem::apply(float4*                 src_vertex,
						 float4*                 dst_vertex,
						 float4*                 dst_normal,
						 float4x4&               deltaTransform,
						 unsigned int            width,
						 unsigned int            height,
						 Matrix6x7f&             res,
						 LinearSystemConfidence& confidence)
{
	const unsigned int localWindowSize = 24;
	const unsigned int blockSize = 64;
	const unsigned int dimX = (unsigned int)ceil(((float)width*height) / (localWindowSize*blockSize));

	launch_build_linear_system(width, height, d_output, src_vertex, dst_vertex, dst_normal, deltaTransform, localWindowSize, blockSize);
	checkCudaErrors(cudaMemcpy(h_output, d_output, sizeof(float)* 30 * dimX, cudaMemcpyDeviceToHost));

	// Copy to CPU
	res = reductionSystem(h_output, dimX, confidence);
}

Matrix6x7f LinearSystem::reductionSystem(const float* data, unsigned int nElems, LinearSystemConfidence& confidence)
{
	Matrix6x7f res;
	res.setZero();

	confidence.reset();

	float numCorrF = 0.0f;

	for (unsigned int k = 0; k < nElems; k++)
	{
		unsigned int linRowStart = 0;

		for (unsigned int i = 0; i < 6; i++)
		{
			for (unsigned int j = i; j < 6; j++)
			{
				res(i, j) += data[30 * k + linRowStart + j - i];
			}

			linRowStart += 6 - i;

			res(i, 6) += data[30 * k + 21 + i];
		}

		confidence.sumRegError += data[30 * k + 27];
		confidence.sumRegWeight += data[30 * k + 28];

		numCorrF += data[30 * k + 29];
	}

	// Fill lower triangle
	for (unsigned int i = 0; i < 6; i++)
	{
		for (unsigned int j = i; j < 6; j++)
		{
			res(j, i) = res(i, j);
		}
	}

	confidence.numCorr = (unsigned int)numCorrF;

	return res;
}

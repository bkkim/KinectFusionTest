#include "SensorUtil.h"
#include <string>

namespace util
{
	
	inline float gaussD(float sigma, int x, int y)
	{
		return exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
	}

	inline float gaussR(float sigma, float dist)
	{
		return exp(-(dist*dist) / (2.0*sigma*sigma));
	}

	void bilateral_filter(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
	{
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++)
			{
				const int kernelRadius = (int)ceil(2.0*sigmaD);

				d_output[y*width + x] = MINF;

				float sum = 0.0f;
				float sumWeight = 0.0f;

				const float depthCenter = d_input[y*width + x];
				if (depthCenter != MINF)
				{
					for (int m = x - kernelRadius; m <= x + kernelRadius; m++) {
						for (int n = y - kernelRadius; n <= y + kernelRadius; n++) {

							if (m >= 0 && n >= 0 && m < width && n < height)
							{
								const float currentDepth = d_input[n*width + m];

								if (currentDepth != MINF) {
									const float weight = gaussD(sigmaD, m - x, n - y)*gaussR(sigmaR, currentDepth - depthCenter);

									sumWeight += weight;
									sum += weight*currentDepth;
								}
							}
						}
					}

					if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
				}
			}
		}
	}

	void gauss_filter(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
	{
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++)
			{
				const int kernelRadius = (int)ceil(2.0*sigmaD);

				d_output[y*width + x] = MINF;

				float sum = 0.0f;
				float sumWeight = 0.0f;

				const float depthCenter = d_input[y*width + x];
				if (depthCenter != MINF)
				{
					for (int m = x - kernelRadius; m <= x + kernelRadius; m++) {
						for (int n = y - kernelRadius; n <= y + kernelRadius; n++) {

							if (m >= 0 && n >= 0 && m < width && n < height)
							{
								const float currentDepth = d_input[n*width + m];

								if (currentDepth != MINF && fabs(depthCenter-currentDepth) < sigmaR) 
								{
									const float weight = gaussD(sigmaD, m - x, n - y);

									sumWeight += weight;
									sum += weight*currentDepth;
								}
							}
						}
					}

					if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
				}
			}
		}
	}

	void subtract_foreground(float* d_output, float* d_input, float foreground_depth_min, float foreground_depth_max, unsigned int width, unsigned int height)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int offset = y * width + x;

				if (d_input[offset] < foreground_depth_min || d_input[offset] > foreground_depth_max)
					d_output[offset] = MINF;
				else
					d_output[offset] = d_input[offset];
			}
		}
	}

	void convert_depth2cameraspace(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				d_output[y*width + x] = make_float4(MINF, MINF, MINF, MINF);

				float depth = d_input[y*width + x];

				if (depth != MINF)
				{
					float4 cameraSpace(intrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, 1.0f));
					d_output[y*width + x] = make_float4(cameraSpace.x, cameraSpace.y, cameraSpace.z, 1.0f);
				}
			}
		}
	}

	void compute_normals(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				d_output[y*width + x] = make_float4(MINF, MINF, MINF, MINF);

				if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
				{
					const float4 CC = d_input[(y + 0)*width + (x + 0)];
					const float4 PC = d_input[(y + 1)*width + (x + 0)];
					const float4 CP = d_input[(y + 0)*width + (x + 1)];
					const float4 MC = d_input[(y - 1)*width + (x + 0)];
					const float4 CM = d_input[(y + 0)*width + (x - 1)];

					if (CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
					{
						const float3 n = cross(make_float3(PC) - make_float3(MC), make_float3(CP) - make_float3(CM));
						const float  l = length(n);

						if (l > 0.0f)
						{
							d_output[y*width + x] = make_float4(n / -l, 1.0f);
						}
					}
				}
			}
		}
	}


}
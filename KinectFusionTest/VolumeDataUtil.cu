
#include "VolumeData.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

extern "C" void launch_reset(uint3 voxel_dim, float* tsdf, float* tsdf_weight, uchar3* color, uchar* color_weight)
{
	uint volume_size = voxel_dim.x * voxel_dim.y * voxel_dim.z;
	float fill_value = 1.0f;
	
	// wrap up the pointer in a device_ptr
	thrust::device_ptr<float> dev_ptr(tsdf);
	thrust::fill(dev_ptr, dev_ptr + volume_size, fill_value);

	checkCudaErrors(cudaMemset(tsdf_weight,  0x00, sizeof(float )*volume_size));
	checkCudaErrors(cudaMemset(color,        0x00, sizeof(uchar3)*volume_size));
	checkCudaErrors(cudaMemset(color_weight, 0x00, sizeof(uchar )*volume_size));
}


#include "TSDFUpdate.h"

#define T_PER_BLOCK 16

namespace tsdf 
{
	namespace cuda
	{
		__global__ 
		void tsdf_update_kernel(
			float4x4 intrinsic, 
			float4x4 extrinsic,
			uint     width, 
			uint     height, 
			float    *img_depth,    // [IN]
			uchar3   *img_color,    // [IN]
			uint3    voxel_dim, 
			float3   voxel_origin, 
			float    voxel_size, 
			float    trunc_margin,
			float    *tsdf,         // [OUT]
			float    *tsdf_weight,  // [OUT]
			uchar3   *color,        // [OUT]
			uchar    *color_weight) // [OUT]
		{
			const int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < voxel_dim.x * voxel_dim.y * voxel_dim.z)
			{
				uint3 s_idx;
				s_idx.z = (idx / (voxel_dim.x * voxel_dim.y));
				s_idx.y = (idx % (voxel_dim.x * voxel_dim.y)) / voxel_dim.x;
				s_idx.x = (idx % (voxel_dim.x * voxel_dim.y)) % voxel_dim.x;

				float3 vertex;
				vertex.x = voxel_origin.x + ((static_cast<float>(s_idx.x) + 0.5f) * voxel_size);
				vertex.y = voxel_origin.y + ((static_cast<float>(s_idx.y) + 0.5f) * voxel_size);
				vertex.z = voxel_origin.z + ((static_cast<float>(s_idx.z) + 0.5f) * voxel_size);

				// Convert world coordinates to camera coordinates
				float3 cam_space = extrinsic * vertex;
				if (cam_space.z <= 0)
					return;

				float3 img_coord = intrinsic * cam_space;
				uint u = roundf(img_coord.x / img_coord.z);
				uint v = roundf(img_coord.y / img_coord.z);
				if (u < 0 || u >= width || v < 0 || v >= height)
					return;

				float depth_val = img_depth[v * width + u];
				if (depth_val == MINF)
					return;

				float diff = (depth_val - cam_space.z);
				if (diff <= -trunc_margin)
					return;

				// TSDF update
				float dist       = fmin(1.0f, diff / trunc_margin);
				float weight_old = tsdf_weight[idx];
				float weight_new = weight_old + 1.0f;// (1.0f - abs(dist));
				uchar alpha      = (uchar)((1 - abs(dist)) * 255);
				if (weight_new > 150.0f)
					weight_new = 150.0f;

				tsdf        [idx] = (tsdf[idx] * weight_old + dist) / weight_new;
				tsdf_weight [idx] = weight_new;
				color       [idx] = img_color[v*width + u];
				color_weight[idx] = (uchar)((color_weight[idx] * weight_old + alpha) / weight_new);
			}
		}

		void update(
			float4x4 intrinsic, 
			float4x4 extrinsic,
			uint     width, 
			uint     height, 
			float    *img_depth, 
			uchar3   *img_color,
			uint3    voxel_dim, 
			float3   voxel_origin, 
			float    voxel_size, 
			float    trunc_margin,
			float    *tsdf,
			float    *tsdf_weight,
			uchar3   *color, 
			uchar    *color_weight)
		{
			uint sample_count = voxel_dim.x * voxel_dim.y * voxel_dim.z;
			const dim3 grid((sample_count + T_PER_BLOCK*T_PER_BLOCK - 1) / (T_PER_BLOCK*T_PER_BLOCK));
			const dim3 block(T_PER_BLOCK*T_PER_BLOCK);

			tsdf_update_kernel <<< grid, block >>> (
				intrinsic, 
				extrinsic,
				width, 
				height, 
				img_depth, 
				img_color, 
				voxel_dim, 
				voxel_origin, 
				voxel_size, 
				trunc_margin,
				tsdf, 
				tsdf_weight, 
				color, 
				color_weight);
#ifdef _DEBUG
			checkCudaErrors(cudaDeviceSynchronize());
			getLastCudaError(__FUNCTION__);
#endif
		}
	}
}

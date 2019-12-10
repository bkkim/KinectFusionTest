
#include "TSDFRaycast.h"

#define T_PER_BLOCK 16

namespace tsdf {
	namespace cuda {

		__device__ 
		void trilinearly_interpolation()
		{

		}

		__global__ 
		void depth_raycast_kernel(
			float4x4 intrinsic,
			float4x4 extrinsic,
			uint     width, 
			uint     height,
			float3   volume_origin, 
			uint3    volume_dim, 
			float    voxel_size, 
			float    trunc_margin,
			float    *tsdf, 
			float    *weight, 
			float    *raycast_depth,
			float4   *raycast_vertex,
			float4   *raycast_normal,
			uchar4   *raycast_color)
		{
			const int x = blockIdx.x * blockDim.x + threadIdx.x;
			const int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x >= width || y >= height)
				return;

			raycast_depth [y*width + x] = MINF;
			raycast_vertex[y*width + x] = make_float4(MINF, MINF, MINF, MINF);
			raycast_normal[y*width + x] = make_float4(MINF, MINF, MINF, MINF);
			raycast_color [y*width + x] = make_uchar4(0, 0, 0, 0);

			float ray_length = 0.5;   // voxel_origin.z;// fx;
			float delta_t    = 0.001; // voxel_size;
			float tsdf_prev  = 0.f;

			// Pixel to Camera
			float3 pt_cam = intrinsic.getInverse() * make_float3(x, y, 1);

			// Camera to world
			float3 ray_dir = extrinsic.getFloat3x3() * pt_cam;

			float3 translation;
			translation.x = extrinsic.m14;
			translation.y = extrinsic.m24;
			translation.z = extrinsic.m34;

			//! Ray trace
			while (true)
			{
				// The maximum length of the ray is 2m
				ray_length += delta_t;
				if (ray_length > 2.f) return;

				float3 pt_grid = ((translation + (ray_dir*ray_length)) / voxel_size) + (-1.f*(volume_origin / voxel_size));

				float u = modf(pt_grid.x, &pt_grid.x);
				float v = modf(pt_grid.y, &pt_grid.y);
				float w = modf(pt_grid.z, &pt_grid.z);

				// Check the ray, whether it is in the view frustrum.
				if (pt_grid.x < 0 || pt_grid.x >= volume_dim.x - 1 ||
					pt_grid.y < 0 || pt_grid.y >= volume_dim.y - 1 ||
					pt_grid.z < 0 || pt_grid.z >= volume_dim.z - 1)
					continue;
				
				int step_z = volume_dim.x*volume_dim.y;
				int step_y = volume_dim.x;
				
				int p000 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x  );    if (weight[p000] == 0) continue;
				int p001 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x  );    if (weight[p001] == 0) continue;
				int p010 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x  );    if (weight[p010] == 0) continue;
				int p011 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x  );    if (weight[p011] == 0) continue;
				int p100 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x+1);    if (weight[p100] == 0) continue;
				int p101 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x+1);    if (weight[p101] == 0) continue;
				int p110 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x+1);    if (weight[p110] == 0) continue;
				int p111 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x+1);    if (weight[p111] == 0) continue;

				// Trilinear interpolation of the tsdf value.
				float tsdf_curr = (1-u)*(1-v)*(1-w)*tsdf[p000]
								+ (1-u)*(1-v)*(  w)*tsdf[p001]
								+ (1-u)*(  v)*(1-w)*tsdf[p010]
								+ (1-u)*(  v)*(  w)*tsdf[p011]
								+ (  u)*(1-v)*(1-w)*tsdf[p100]
								+ (  u)*(1-v)*(  w)*tsdf[p101]
								+ (  u)*(  v)*(1-w)*tsdf[p110]
								+ (  u)*(  v)*(  w)*tsdf[p111];

				// occlusion area
				if (tsdf_prev < 0.f && tsdf_curr > 0.f)
					break;
				// zero crossing
				if (tsdf_prev > 0.f && tsdf_curr < 0.f) {
					raycast_depth[y*width + x] = ray_length + (tsdf_curr*trunc_margin);

					const float t_star = ray_length - trunc_margin * 0.5f * tsdf_prev / (tsdf_curr - tsdf_prev);
					const auto vertex = translation + ray_dir * t_star;
					raycast_vertex[y*width + x] = make_float4(vertex, 1.0f);


					raycast_normal[y*width + x];
					break;
				}
				// free space
				else {
					tsdf_prev = tsdf_curr;
				}
			}
			
		}

		void raycast(float4x4 intrinsic, 
			         float4x4 extrinsic,
			         uint     width, 
			         uint     height,
			         float3   volume_origin, 
			         uint3    volume_dim, 
			         float    voxel_size, 
			         float    trunc_margin,
			         float    *tsdf, 
			         float    *tsdf_weight, 
			         float    *raycast_depth,
			         float4   *raycast_vertex,
			         float4   *raycast_normal,
			         uchar4   *raycast_color)
		{
			const dim3 grid((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
			const dim3 block(T_PER_BLOCK, T_PER_BLOCK);

			depth_raycast_kernel <<< grid, block >>> (intrinsic, 
				                                      extrinsic,
				                                      width, 
				                                      height,
				                                      volume_origin, 
				                                      volume_dim, 
				                                      voxel_size, 
				                                      trunc_margin,
				                                      tsdf, 
				                                      tsdf_weight, 
				                                      raycast_depth,
				                                      raycast_vertex,
				                                      raycast_normal,
				                                      raycast_color);
#ifdef _DEBUG
			checkCudaErrors(cudaDeviceSynchronize());
			getLastCudaError(__FUNCTION__);
#endif
		}
	}
}
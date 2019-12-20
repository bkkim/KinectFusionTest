
#include "TSDFRaycast.h"

#define T_PER_BLOCK 16

namespace tsdf {
	namespace cuda {

		__device__ 
		bool trilinear_interpolation(uint3 volume_dim, float3 pt_grid, float* tsdf, float* weight, float& ret)
		{
			int step_z = volume_dim.x*volume_dim.y;
			int step_y = volume_dim.x;

			float u = modf(pt_grid.x, &pt_grid.x);
			float v = modf(pt_grid.y, &pt_grid.y);
			float w = modf(pt_grid.z, &pt_grid.z);

			int p000 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x  );    if (weight[p000] == 0) return false;
			int p001 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x  );    if (weight[p001] == 0) return false;
			int p010 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x  );    if (weight[p010] == 0) return false;
			int p011 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x  );    if (weight[p011] == 0) return false;
			int p100 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x+1);    if (weight[p100] == 0) return false;
			int p101 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y  )* step_y + ((int)pt_grid.x+1);    if (weight[p101] == 0) return false;
			int p110 = ((int)pt_grid.z  )* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x+1);    if (weight[p110] == 0) return false;
			int p111 = ((int)pt_grid.z+1)* step_z + ((int)pt_grid.y+1)* step_y + ((int)pt_grid.x+1);    if (weight[p111] == 0) return false;
			
			// Trilinear interpolation of the tsdf value.
			ret = (1-u)*(1-v)*(1-w)*tsdf[p000]
				+ (1-u)*(1-v)*(  w)*tsdf[p001]
				+ (1-u)*(  v)*(1-w)*tsdf[p010]
				+ (1-u)*(  v)*(  w)*tsdf[p011]
				+ (  u)*(1-v)*(1-w)*tsdf[p100]
				+ (  u)*(1-v)*(  w)*tsdf[p101]
				+ (  u)*(  v)*(1-w)*tsdf[p110]
				+ (  u)*(  v)*(  w)*tsdf[p111];
			return true;
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

			float ray_length = 0.5f;   // voxel_origin.z;// fx;
			float delta_t    = 0.001f; // voxel_size;
			float tsdf_curr  = 0.0f;
			float tsdf_prev  = 0.0f;
			float max_length = 2.0f;

			// Pixel to Camera
			float3 pt_cam = intrinsic.getInverse() * make_float3(x, y, 1);

			// Camera to world
			float3 ray_dir = extrinsic.getFloat3x3() * pt_cam;

			float3 translation;
			translation.x = extrinsic.m14;
			translation.y = extrinsic.m24;
			translation.z = extrinsic.m34;

			//! Ray trace
			for (;ray_length < max_length; ray_length += delta_t)
			{
				float3 pt      = (translation + (ray_dir*ray_length));
				float3 pt_grid = (pt + (-1.f*volume_origin)) / voxel_size;

				// Check the ray, whether it is in the view frustrum.
				if (pt_grid.x < 0 || pt_grid.x >= volume_dim.x - 1 ||
					pt_grid.y < 0 || pt_grid.y >= volume_dim.y - 1 ||
					pt_grid.z < 0 || pt_grid.z >= volume_dim.z - 1)
					continue;

				//int step_z = volume_dim.x*volume_dim.y;
				//int step_y = volume_dim.x;
				//float tsdf_curr = tsdf[(int)pt_grid.z*step_z + (int)pt_grid.y*step_y + (int)pt_grid.x];

				if (!trilinear_interpolation(volume_dim, pt_grid, tsdf, weight, tsdf_curr))
					continue;
				
				// occlusion area
				if (tsdf_prev < 0.f && tsdf_curr > 0.f)
					break;
				// zero crossing
				if (tsdf_prev > 0.f && tsdf_curr < 0.f) 
				{
					//float t_star = ray_length - ((delta_t * tsdf_prev) / (tsdf_curr - tsdf_prev));
					//float3 vertex = translation + ray_dir * t_star;
				
					float depth = ray_length + (tsdf_curr*trunc_margin);
					raycast_depth [y*width + x] = depth;
					raycast_vertex[y*width + x] = make_float4(pt_cam*depth, 1.0f);

					float3 normal, shifted;
					float  fx1, fx2, fy1, fy2, fz1, fz2;

					shifted = pt_grid;
					shifted.x += 1;
					if (shifted.x >= volume_dim.x - 1)
						break;
					if (!trilinear_interpolation(volume_dim, shifted, tsdf, weight, fx1)) break;

					shifted = pt_grid;
					shifted.x -= 1;
					if (shifted.x < 1)
						break;
					if (!trilinear_interpolation(volume_dim, shifted, tsdf, weight, fx2)) break;

					normal.x = (fx1 - fx2);

					shifted = pt_grid;
					shifted.y += 1;
					if (shifted.y >= volume_dim.y - 1)
						break;
					if (!trilinear_interpolation(volume_dim, shifted, tsdf, weight, fy1)) break;

					shifted = pt_grid;
					shifted.y -= 1;
					if (shifted.y < 1)
						break;
					if (!trilinear_interpolation(volume_dim, shifted, tsdf, weight, fy2)) break;

					normal.y = (fy1 - fy2);

					shifted = pt_grid;
					shifted.z += 1;
					if (shifted.z >= volume_dim.z - 1)
						break;
					if (!trilinear_interpolation(volume_dim, shifted, tsdf, weight, fz1)) break;

					shifted = pt_grid;
					shifted.z -= 1;
					if (shifted.z < 1)
						break;
					if (!trilinear_interpolation(volume_dim, shifted, tsdf, weight, fz2)) break;

					normal.z = (fz1 - fz2);

					if (length(normal) == 0)
						break;

					normal = extrinsic.getFloat3x3().getInverse()*normal;
					normal = normalize(normal);
					raycast_normal[y*width + x] = make_float4(-normal, 0.0f);
					//raycast_color[y*width + x] = ;
					break;
				}
				// free space
				else {
					tsdf_prev = tsdf_curr;
				}
			}
			
		}
		//*/

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
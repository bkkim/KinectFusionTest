
#include "TSDFRaycast.h"

#define T_PER_BLOCK 16

namespace tsdf {
	namespace cuda {

		__global__ 
		void depth_raycast_kernel(
			float4x4 intrinsic,
			float4x4 extrinsic,
			uint     width, 
			uint     height,
			float3   voxel_origin, 
			uint3    voxel_dim, 
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

			float curr_z = 0.5;// voxel_origin.z;// fx;
			float delta_t = 0.001;// voxel_size;
			float tsdf_prev = 0.f;

			//! Ray trace
			while (true)
			{
				// The maximum length of the ray is 2m
				curr_z += delta_t;
				if (curr_z > 2.f) return;

				// Pixel to Camera
				float3 pt_cam = intrinsic.getInverse() * make_float3(x, y, 1) *curr_z;

				// Camera to world
				float3 pt_wld = extrinsic.getInverse() * pt_cam;


				// Find the 8-voxels' indices around the current ray
				float3 pt_grid = (pt_wld / voxel_size) + (-1.f*(voxel_origin / voxel_size));

				float u = modf(pt_grid.x, &pt_grid.x);
				float v = modf(pt_grid.y, &pt_grid.y);
				float w = modf(pt_grid.z, &pt_grid.z);

				// Check the ray, whether it is in the view frustrum.
				if (pt_grid.x < 0 || pt_grid.x >= voxel_dim.x - 1 ||
					pt_grid.y < 0 || pt_grid.y >= voxel_dim.y - 1 ||
					pt_grid.z < 0 || pt_grid.z >= voxel_dim.z - 1)
					continue;
				
				int step_z = voxel_dim.x*voxel_dim.y;
				int step_y = voxel_dim.x;
				
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
					raycast_depth[y*width + x] = curr_z + (tsdf_curr*trunc_margin);
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
			         float3   voxel_grid_origin, 
			         uint3    voxel_grid_dim, 
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
				                                      voxel_grid_origin, 
				                                      voxel_grid_dim, 
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
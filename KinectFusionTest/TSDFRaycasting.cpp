
#include "TSDFRaycasting.h"

namespace tsdf
{
	void raycasting(float4x4 intrinsics,  float4x4 extrinsics,
					uint width, uint height, 
					float3 voxel_origin,  uint3 voxel_dim, float voxel_size, float trunc_margin, 
					float zoom,
					float* tsdf, float* weight, 
					float* out_depth, float4* out_depth4)
	{
		float fx = intrinsics.m11 * 0.001;
		float fy = intrinsics.m22 * 0.001;
		float cx = intrinsics.m13;
		float cy = intrinsics.m23;
	
		int cntLoop = 0;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				out_depth[y*width + x] = MINF;
				out_depth4[y*width + x] = make_float4(MINF, MINF, MINF, MINF);
	
				float curr_z = voxel_origin.z;// fx;
				float delta_t = 0.001;// voxel_size;
	
				bool find_surface = false;
				bool exceed_voxel = false;
	
				float tsdf_prev = 0.f;
				float surface = 1.f;
	
				cntLoop = 0;
				//! Ray trace
				while (!find_surface && !exceed_voxel)
				{
					// Ray's length is voxel_size * 300 (if the voxel size is 1mm, ray's length is 300mm.) 
					if (cntLoop++ > 3000)
					{
						exceed_voxel = true;
						continue;
					}
					curr_z += delta_t;
	
					// Pixel to Camera
					float3 pt_cam = { (((x - cx)*0.001 / zoom) / fx) * curr_z,
									  (((y - cy)*0.001 / zoom) / fy) * curr_z,
									  curr_z };
	
					// Camera to World
					// P_g = Tg,k * P_k
					float3 pt_base = extrinsics * pt_cam;
	
					float3 cube_size = {voxel_dim.x * voxel_size,
										voxel_dim.y * voxel_size,
										voxel_dim.z * voxel_size};
	
					// find the 8-voxels' indices around the current ray
					//float3 pt_grid = {	(pt_base.x + abs(voxel_origin.x)) / voxel_size,
					//					(pt_base.y + abs(voxel_origin.y)) / voxel_size,
					//					(pt_base.z + abs(voxel_origin.z)) / voxel_size };
					float3 pt_grid = {	(pt_base.x + abs(voxel_origin.x)) / voxel_size,
										(pt_base.y + abs(voxel_origin.y)) / voxel_size,
										(pt_base.z - (voxel_origin.z))    / voxel_size };
	
					float u = modf(pt_grid.x, &pt_grid.x);
					float v = modf(pt_grid.y, &pt_grid.y);
					float w = modf(pt_grid.z, &pt_grid.z);
	
					if (pt_grid.x < 0 || pt_grid.x >= voxel_dim.x - 1 ||
						pt_grid.y < 0 || pt_grid.y >= voxel_dim.y - 1 ||
						pt_grid.z < 0 || pt_grid.z >= voxel_dim.z - 1)
					{
						continue;
					}
	
					int p000 = ((int)pt_grid.z)  *voxel_dim.y*voxel_dim.x + ((int)pt_grid.y)  *voxel_dim.x + ((int)pt_grid.x);		if (weight[p000] == 0) continue;
					int p001 = ((int)pt_grid.z+1)*voxel_dim.y*voxel_dim.x + ((int)pt_grid.y)  *voxel_dim.x + ((int)pt_grid.x);		if (weight[p001] == 0) continue;
					int p010 = ((int)pt_grid.z)  *voxel_dim.y*voxel_dim.x + ((int)pt_grid.y+1)*voxel_dim.x + ((int)pt_grid.x);		if (weight[p010] == 0) continue;
					int p100 = ((int)pt_grid.z)  *voxel_dim.y*voxel_dim.x + ((int)pt_grid.y)  *voxel_dim.x + ((int)pt_grid.x+1);	if (weight[p100] == 0) continue;
					int p101 = ((int)pt_grid.z+1)*voxel_dim.y*voxel_dim.x + ((int)pt_grid.y)  *voxel_dim.x + ((int)pt_grid.x+1);	if (weight[p101] == 0) continue;
					int p011 = ((int)pt_grid.z+1)*voxel_dim.y*voxel_dim.x + ((int)pt_grid.y+1)*voxel_dim.x + ((int)pt_grid.x);		if (weight[p011] == 0) continue;
					int p110 = ((int)pt_grid.z)  *voxel_dim.y*voxel_dim.x + ((int)pt_grid.y+1)*voxel_dim.x + ((int)pt_grid.x+1);	if (weight[p110] == 0) continue;
					int p111 = ((int)pt_grid.z+1)*voxel_dim.y*voxel_dim.x + ((int)pt_grid.y+1)*voxel_dim.x + ((int)pt_grid.x+1);	if (weight[p111] == 0) continue;
	
					//
					float tsdf_curr = (1-u)*(1-v)*(1-w)*tsdf[p000]
									+ (1-u)*(1-v)*(w)  *tsdf[p001]
									+ (1-u)*(v)  *(1-w)*tsdf[p010]
									+ (u)  *(1-v)*(1-w)*tsdf[p100]
									+ (u)  *(1-v)*(w)  *tsdf[p101]
									+ (1-u)*(v)  *(w)  *tsdf[p011]
									+ (u)  *(v)  *(1-w)*tsdf[p110]
									+ (u)  *(v)  *(w)  *tsdf[p111];
	
					if (tsdf_prev < 0.f && tsdf_curr > 0.f)
						break;
					else
					if (tsdf_prev > 0.f && tsdf_curr < 0.f)	// zero crossing
					{
						find_surface = true;
	
						float depth = pt_cam.z + (tsdf_curr*trunc_margin);
						out_depth[y*width + x] = depth;
						out_depth4[y*width + x] = make_float4(pt_cam, 1.0f);
	
						//uchar4 color_val;
						//color_val.z = (1-u)*(1-v)*(1-w)*tsdf_clr[p000].x
						//			+ (1-u)*(1-v)*(w)  *tsdf_clr[p001].x
						//			+ (1-u)*(v)  *(1-w)*tsdf_clr[p010].x
						//			+ (u)  *(1-v)*(1-w)*tsdf_clr[p100].x
						//			+ (u)  *(1-v)*(w)  *tsdf_clr[p101].x
						//			+ (1-u)*(v)  *(w)  *tsdf_clr[p011].x
						//			+ (u)  *(v)  *(1-w)*tsdf_clr[p110].x
						//			+ (u)  *(v)  *(w)  *tsdf_clr[p111].x;
						//color_val.y = (1-u)*(1-v)*(1-w)*tsdf_clr[p000].y
						//			+ (1-u)*(1-v)*(w)  *tsdf_clr[p001].y
						//			+ (1-u)*(v)  *(1-w)*tsdf_clr[p010].y
						//			+ (u)  *(1-v)*(1-w)*tsdf_clr[p100].y
						//			+ (u)  *(1-v)*(w)  *tsdf_clr[p101].y
						//			+ (1-u)*(v)  *(w)  *tsdf_clr[p011].y
						//			+ (u)  *(v)  *(1-w)*tsdf_clr[p110].y
						//			+ (u)  *(v)  *(w)  *tsdf_clr[p111].y;
						//color_val.x = (1-u)*(1-v)*(1-w)*tsdf_clr[p000].z
						//			+ (1-u)*(1-v)*(w)  *tsdf_clr[p001].z
						//			+ (1-u)*(v)  *(1-w)*tsdf_clr[p010].z
						//			+ (u)  *(1-v)*(1-w)*tsdf_clr[p100].z
						//			+ (u)  *(1-v)*(w)  *tsdf_clr[p101].z
						//			+ (1-u)*(v)  *(w)  *tsdf_clr[p011].z
						//			+ (u)  *(v)  *(1-w)*tsdf_clr[p110].z
						//			+ (u)  *(v)  *(w)  *tsdf_clr[p111].z;
						//color_val.w = (1-u)*(1-v)*(1-w)*tsdf_clr[p000].w
						//			+ (1-u)*(1-v)*(w)  *tsdf_clr[p001].w
						//			+ (1-u)*(v)  *(1-w)*tsdf_clr[p010].w
						//			+ (u)  *(1-v)*(1-w)*tsdf_clr[p100].w
						//			+ (u)  *(1-v)*(w)  *tsdf_clr[p101].w
						//			+ (1-u)*(v)  *(w)  *tsdf_clr[p011].w
						//			+ (u)  *(v)  *(1-w)*tsdf_clr[p110].w
						//			+ (u)  *(v)  *(w)  *tsdf_clr[p111].w;
						////color_val.w = 255;
						//out_color[y*width + x] = color_val;
					}
					else
					{
						tsdf_prev = tsdf_curr;
						surface = pt_base.z;
					}
				}
			}
		}
	}

}
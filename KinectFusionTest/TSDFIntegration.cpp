#include "TSDFIntegration.h"

namespace tsdf 
{
	void integration(float4x4	intrinsic,
					 float4x4	extrinsic,
					 uint		width,
					 uint		height,
					 float		*img_depth,
					 uchar3		*img_color,
					 uint3		voxel_dim,
					 float3		voxel_origin,
					 float		voxel_size,
					 float		trunc_margin,
					 float		*tsdf,
					 float		*tsdf_weight,
					 uchar3		*color,
					 uchar		*color_weight)
	{
		float fx = intrinsic.m11;
		float fy = intrinsic.m22;
		float cx = intrinsic.m13;
		float cy = intrinsic.m23;

		for (int idx_z = 0; idx_z < voxel_dim.z; ++idx_z) {
			for (int idx_y = 0; idx_y < voxel_dim.y; ++idx_y) {
				for (int idx_x = 0; idx_x < voxel_dim.x; ++idx_x) 
				{
					// Convert voxel center from grid coordinates to base frame camera coordinates
					float base_x = voxel_origin.x + idx_x * voxel_size;
					float base_y = voxel_origin.y + idx_y * voxel_size;
					float base_z = voxel_origin.z + idx_z * voxel_size;

					// Convert from base frame camera coordinates to current frame camera coordinates
					// Pc = iR(Pw-t)
					float tmp_pt[3] = { 0 };
					tmp_pt[0] = base_x - extrinsic.m14 * 0.001;
					tmp_pt[1] = base_y - extrinsic.m24 * 0.001;
					tmp_pt[2] = base_z - extrinsic.m34 * 0.001;
					float cam_x = extrinsic.m11 * tmp_pt[0] + extrinsic.m21 * tmp_pt[1] + extrinsic.m31 * tmp_pt[2];
					float cam_y = extrinsic.m12 * tmp_pt[0] + extrinsic.m22 * tmp_pt[1] + extrinsic.m32 * tmp_pt[2];
					float cam_z = extrinsic.m13 * tmp_pt[0] + extrinsic.m23 * tmp_pt[1] + extrinsic.m33 * tmp_pt[2];

				//	if (cam_z <= 0)
				//		continue;

					// camera coordinate to pixel coordinate
					int pix_u = roundf(fx * (cam_x / cam_z) + cx);
					int pix_v = roundf(fy * (cam_y / cam_z) + cy);
					if (pix_u < 0 || pix_u >= width || pix_v < 0 || pix_v >= height)
						continue;

					float depth_val = img_depth[pix_v * width + pix_u];
					if (depth_val <= 0 || depth_val > 6)
						continue;

				//	float diff = (depth_val - cam_z);
					float diff = sqrtf(1 + powf((cam_x / cam_z), 2) + powf((cam_y / cam_z), 2));
					if (diff <= -trunc_margin || diff >= trunc_margin)
						continue;

					// Integrate
					uint volume_idx = (idx_z*voxel_dim.y*voxel_dim.x) + (idx_y*voxel_dim.x) + idx_x;
					float dist = fmin(1.0f, diff / trunc_margin);
					float weight_old = tsdf_weight[volume_idx];
					float weight_new = weight_old + 1.0f;
					tsdf_weight[volume_idx] = weight_new;
					tsdf[volume_idx] = (tsdf[volume_idx] * weight_old + dist) / weight_new;

					color[volume_idx] = img_color[pix_v * width + pix_u];
					uchar alpha = (uchar)((1 - abs(dist)) * 255);
					color_weight[volume_idx] = (uchar)((color_weight[volume_idx] * weight_old + alpha) / weight_new);
				}
			}
		}
	}

}
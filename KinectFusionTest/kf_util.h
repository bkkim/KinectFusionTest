#pragma once

///
// For voxel debugging
void ShowTSDFCuttingPlane(uint3 voxel_grid_dim, float* gpu_voxel_grid_tsdf, float* gpu_voxel_grid_weight, uchar3* gpu_voxel_grid_color)
{
	uint volume_size = voxel_grid_dim.x * voxel_grid_dim.y * voxel_grid_dim.z;
	float* voxel_grid_tsdf = new float[volume_size];
	float* voxel_grid_weight = new float[volume_size];
	uchar3* voxel_grid_color = new uchar3[volume_size];

	cudaMemcpy(voxel_grid_tsdf, gpu_voxel_grid_tsdf, sizeof(float)*volume_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, sizeof(float)*volume_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(voxel_grid_color, gpu_voxel_grid_color, sizeof(uchar3)*volume_size, cudaMemcpyDeviceToHost);

	int w = voxel_grid_dim.x;
	int h = voxel_grid_dim.y;

	cv::Mat d_tsdf(h, w, CV_32FC1);
	cv::Mat d_weight(h, w, CV_8UC1);
	cv::Mat d_color(h, w, CV_8UC1);

	for (int d = 0; d < voxel_grid_dim.z; d++) {
		for (int v = 0; v < h; v++) {
			for (int u = 0; u < w; u++) {
				d_tsdf.at<float>(v*w + u) = voxel_grid_tsdf[d*h*w + v*w + u];
				d_weight.at<uchar>(v*w + u) = (uchar)(voxel_grid_weight[d*h*w + v*w + u]);
			}
		}

		memcpy(d_color.data, &voxel_grid_color[d*h*w], sizeof(uchar3)*h*w);

		std::cout << "voxel_grid_dim_z: " << d << std::endl;
		cv::imshow("result tsdf", d_tsdf);
		cv::imshow("result weight", d_weight);
		cv::imshow("result color", d_color);
		cv::waitKey(0);
	}

	delete[] voxel_grid_tsdf;
	delete[] voxel_grid_weight;
	delete[] voxel_grid_color;
}


void ShowDerivedData(int width, int height, float* depth, float* depth_filtered, float* depth_foreground, float4* vertex, float4* normal, uchar3* color)
{
	cv::Mat mat_depth(height, width, CV_32FC1);
	cv::Mat mat_depth_filtered(height, width, CV_32FC1);
	cv::Mat mat_depth_foreground(height, width, CV_32FC1);
	cv::Mat4f mat_vertex(height, width, CV_32FC4);
	cv::Mat4f mat_normal(height, width, CV_32FC4);
	cv::Mat mat_color(height, width, CV_8UC3);

	int size = width * height;
	cudaMemcpy((float*)(mat_depth.data), depth, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy((float*)(mat_depth_filtered.data), depth_filtered, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy((float*)(mat_depth_foreground.data), depth_foreground, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy((float4*)(mat_vertex.data), vertex, sizeof(float4)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy((float4*)(mat_normal.data), normal, sizeof(float4)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy((uchar3*)(mat_color.data), color, sizeof(uchar3)*size, cudaMemcpyDeviceToHost);

	cv::imshow("depth", mat_depth);
	cv::imshow("depth_filtered", mat_depth_filtered);
	cv::imshow("depth_foreground", mat_depth_foreground);
	cv::imshow("vertex", mat_vertex);
	cv::imshow("normal", mat_normal);
	cv::imshow("color", mat_color);
	cv::waitKey(0);

	return;
}

void ExtractSampledVoxel(FILE *fp, float *weight, uint3 voxel_dim, float3 voxel_origin, float voxel_size)
{
	float* h_weight = new float[voxel_dim.x*voxel_dim.y*voxel_dim.z];
	cudaMemcpy(h_weight, weight, sizeof(float)*voxel_dim.x*voxel_dim.y*voxel_dim.z, cudaMemcpyDeviceToHost);

	for (int z = 0; z < voxel_dim.z; z++)
	{
		for (int y = 0; y < voxel_dim.y; y++)
		{
			for (int x = 0; x < voxel_dim.x; x++)
			{
				uint idx = z * voxel_dim.y * voxel_dim.z + y * voxel_dim.x + x;
				if (h_weight[idx] > 0)
				{
					float v_x = voxel_origin.x + (x * voxel_size);
					float v_y = voxel_origin.y + (y * voxel_size);
					float v_z = voxel_origin.z + (z * voxel_size);

					// Voxel indices of x, y, z and vertex's position x, y, z
					fprintf(fp, "%d %d %d %f %f %f\n", x, y, z, v_x, v_y, v_z);
				}
			}
		}
	}

	delete[] h_weight;
}

void ConvertMesh2Ply(FILE *fp, uint verts_count, float3* pos, float3* norm, float3* colors)
{
	float3* verts_pos = new float3[verts_count];
	float3* verts_norm = new float3[verts_count];
	float3* verts_color = new float3[verts_count];
	cudaMemcpy(verts_pos, pos, sizeof(float3)*verts_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(verts_norm, norm, sizeof(float3)*verts_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(verts_color, colors, sizeof(float3)*verts_count, cudaMemcpyDeviceToHost);

	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %d\n", verts_count);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property float nx\n");
	fprintf(fp, "property float ny\n");
	fprintf(fp, "property float nz\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "element face %d\n", verts_count / 3);
	fprintf(fp, "property list uchar int vertex_indices\n");
	fprintf(fp, "property int flags\n");
	fprintf(fp, "end_header\n");

	// Create point cloud content for ply file
	for (int i = 0; i < verts_count; i++)
	{
		//	if (isnan(norm[i].x) || isnan(norm[i].y) || isnan(norm[i].z))
		//		continue;

		fprintf(fp, "%.8f %.8f %.8f %.8f %.8f %.8f %d %d %d \n", verts_pos[i].x, verts_pos[i].y, verts_pos[i].z,
			verts_norm[i].x, verts_norm[i].y, verts_norm[i].z,
			(uchar)(verts_color[i].z * 255), (uchar)(verts_color[i].y * 255), (uchar)(verts_color[i].x * 255));
	}
	// Write the face informations.
	for (int i = 0; i < verts_count; i = i + 3)
	{
		fprintf(fp, "3 %d %d %d 0\n", i, i + 1, i + 2);
	}

	delete[] verts_pos;
}

cv::Mat Mesh2Depth(uint width, uint height, float4x4 intrinsic, float4x4 extrinsic, uint verts_count, float3* pos)
{
	float3* verts_pos = new float3[verts_count];
	cudaMemcpy(verts_pos, pos, sizeof(float3)*verts_count, cudaMemcpyDeviceToHost);

	cv::Mat mat(height, width, CV_32FC1);
	memset((float*)mat.data, 0x00, sizeof(float)*width*height);

	for (int i = 0; i < verts_count; i++)
	{
		float4 target = intrinsic * make_float4(verts_pos[i], 1.0f);
		int x = target.x / target.z;
		int y = target.y / target.z;
		if (x > 0 && x < width &&
			y > 0 && y < height)
		{
			if (mat.at<float>(y, x) == 0x00)
			{
				mat.at<float>(y, x) = (float)(target.z);
			}
			else
			{
				// Set minimum depth value.
				if (mat.at<float>(y, x) > (float)(target.z))
				{
					mat.at<float>(y, x) = (float)(target.z);
				}
			}
		}
	}

	//	cv::imshow("depth", mat);
	//	cv::waitKey(0);

	delete[] verts_pos;
	return mat;
}

cv::Mat Mesh2FullDepth(uint width, uint height, float* data)
{
	float* h_data = new float[width*height];
	cudaMemcpy(h_data, data, sizeof(float)*width*height, cudaMemcpyDeviceToHost);

	cv::Mat mat(height, width, CV_32FC1);
	memcpy((float*)mat.data, h_data, sizeof(float)*width*height);

	delete [] h_data;
	return mat;
}

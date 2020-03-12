
#include "kf_main.h"

CUDARGBDSensor       *g_CUDARGBDSensor  = NULL;
CUDATSDFMerger       *g_CUDATSDFMerger  = NULL;
PointToPlaneICP      *g_PointToPlaneICP = NULL;
CUDATSDFMarchingCube *g_CUDATSDFMarchingCube = NULL;

void initializeAll()
{
	// Set SensorParam
	SensorParam sensorParam;
	sensorParam.width    = INPUT_WIDTH;
	sensorParam.height   = INPUT_HEIGHT;
	sensorParam.depthMin = DEPTH_MIN;
	sensorParam.depthMax = DEPTH_MAX;
	sensorParam.intrinsic.setIdentity();
	sensorParam.intrinsic.m11 = KINECT_FOCAL_X;
	sensorParam.intrinsic.m22 = KINECT_FOCAL_Y;
	sensorParam.intrinsic.m13 = KINECT_CENTER_X;
	sensorParam.intrinsic.m23 = KINECT_CENTER_Y;
	sensorParam.extrinsic.setIdentity();

	// Create CUDARGBDSensor
	RGBDSensor* rgbSensor = new RGBDSensor(sensorParam);
	g_CUDARGBDSensor = new CUDARGBDSensor(rgbSensor);

	// Set FilterParam
	FilterParam filterParam;
	filterParam.sigma_d              = FILTER_SPATIAL;
	filterParam.sigma_r              = FILTER_RANGE;
	filterParam.foreground_depth_min = FILTER_OBJECT_MIN;
	filterParam.foreground_depth_max = FILTER_OBJECT_MAX;
	g_CUDARGBDSensor->setFilterParam(filterParam);

	// Set TSDFVolumeParam
	VolumeParam volumeParam;
	volumeParam.volume_origin = make_float3(VOLUME_ORIGIN_X, VOLUME_ORIGIN_Y, VOLUME_ORIGIN_Z);
	volumeParam.volume_dim    = make_uint3(VOLUME_DIM_X, VOLUME_DIM_Y, VOLUME_DIM_Z);
	volumeParam.voxel_size    = VOXEL_SIZE;
	volumeParam.trunc_margin  = TRUNC_MARGIN;

	// Create CUDATSDFMerger
	g_CUDATSDFMerger = new CUDATSDFMerger(volumeParam, sensorParam);
	g_CUDATSDFMerger->getVolumeData()->reset(volumeParam);

	// Create CUDATSDFMarchingCube
	g_CUDATSDFMarchingCube = new CUDATSDFMarchingCube();
	g_CUDATSDFMarchingCube->create(volumeParam.volume_dim, volumeParam.volume_origin, volumeParam.voxel_size, 0.0f);

	//
	PointToPlaneICP_Params icpParam;
	icpParam.width  = sensorParam.width;
	icpParam.height = sensorParam.height;
	icpParam.fx     = sensorParam.intrinsic.m11;
	icpParam.fy     = sensorParam.intrinsic.m22;
	icpParam.cx     = sensorParam.intrinsic.m13;
	icpParam.cy     = sensorParam.intrinsic.m23;
	icpParam.z_min  = sensorParam.depthMin;
	icpParam.z_max  = sensorParam.depthMax;
	icpParam.level  = ICP_LEVEL;
	icpParam.max_inner_iter      = ICP_MAX_INNER_ITER;
	icpParam.max_outer_iter      = ICP_MAX_OUTER_ITER;
	icpParam.thres_corres_dist   = ICP_THRES_CORRES_DIST;
	icpParam.thres_corres_normal = ICP_THRES_CORRES_NORMAL;
	icpParam.thres_trans_angle   = ICP_THRES_TRANS_ANGLE;
	icpParam.thres_trans_dist    = ICP_THRES_TRANS_DIST;
	icpParam.thres_cond          = ICP_THRES_COND;
	icpParam.early_out           = ICP_EARLY_OUT;

	g_PointToPlaneICP = new PointToPlaneICP(icpParam);
}

void deleteAll()
{
	//
	delete g_CUDARGBDSensor;  g_CUDARGBDSensor  = NULL;
	delete g_CUDATSDFMerger;  g_CUDATSDFMerger  = NULL;
	delete g_PointToPlaneICP; g_PointToPlaneICP = NULL;
}


uint g_idx = 0;

void SaveData(std::ofstream& file, float4* vertex, float4* normal, float4x4 transform, uint width, uint height)
{
	// frame
	float4* h_vertex = new float4[width*height];
	float4* h_normal = new float4[width*height];

	cudaMemcpy(h_vertex, vertex, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_normal, normal, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);

	uint vertex_count = 0;
	uint normal_count = 0;
	for (int i = 0; i < width*height; i++) {
		if (h_vertex[i].x != MINF) vertex_count++;
		if (h_normal[i].x != MINF) normal_count++;
	}

	file << "ply" << std::endl;
	file << "format ascii 1.0" << std::endl;
	file << "element vertex " << vertex_count << std::endl;
	file << "property float x" << std::endl;
	file << "property float y" << std::endl;
	file << "property float z" << std::endl;
	file << "end_header" << std::endl;

	// Create point cloud content for ply file
	for (int i = 0; i < width*height; i++)
	{
		if (h_vertex[i].x == MINF)
			continue;
		file << h_vertex[i].x << " " << h_vertex[i].y << " " << h_vertex[i].z << std::endl;
	}

	SAFE_DELETE(h_vertex);
	SAFE_DELETE(h_normal);

	file.close();
}

int main(int argc, char* argv[])
{
//	float3 xylambda;
//	
//	for (int i = 0; i < INPUT_HEIGHT; i++)
//	{
//		for (int j = 0; j < INPUT_WIDTH; j++)
//		{
//			xylambda.x = ((float)j - (float)KINECT_CENTER_X) / (float)KINECT_FOCAL_X;
//			xylambda.y = ((float)i - (float)KINECT_CENTER_Y) / (float)KINECT_FOCAL_Y;
//			xylambda.z = 1.f;
//			float len = length(xylambda);
//			std::cout << "len: " << len << std::endl;
//		}
//		std::cout << std::endl;
//	}
	

	// Initialize all global variables
	initializeAll();

	int w = INPUT_WIDTH;
	int h = INPUT_HEIGHT;
	int size = w * h;
	float4* h_frame_normal = new float4[size];
	float4* h_model_normal = new float4[size];

	cv::Mat4f mFrameNormal(h, w, CV_32FC4);
	cv::Mat4f mModelNormal(h, w, CV_32FC4);
	cv::Mat4f mBlendNormal(h, w, CV_32FC4);

	bool bFirst = true;
	int idx_start = 900;// 80;// 900;
	int frame_count = 600;// 320;// 600;

	float4x4 current_pose;
	current_pose.setIdentity();

	std::string data_path("../Dataset/Kinect1/ubody180/source");
	char depth_file[MAX_PATH] = {0,};
	char color_file[MAX_PATH] = {0,};

	GpuTimer t_timer;

	for (int idx = idx_start; idx < idx_start + frame_count; idx++)
	{
		std::cout << "Frame idx: " << std::setw(10) << std::left << idx << std::endl;
		memset(depth_file, 0x00, sizeof(char)*MAX_PATH);
		memset(color_file, 0x00, sizeof(char)*MAX_PATH);
		sprintf(depth_file, "%s/depth/d_%d.png", data_path.c_str(), idx);
		sprintf(color_file, "%s/color/c_%d.png", data_path.c_str(), idx);

		cv::Mat color = cv::imread(color_file);
		cv::Mat depth = cv::imread(depth_file, cv::IMREAD_ANYDEPTH);

		//t_timer.Start();
		g_CUDARGBDSensor->process((uchar3*)color.data, (ushort*)depth.data);
		//t_timer.Stop();
		//std::cout << "CUDARGBDSensor    process Time: " << t_timer.Elapsed() << " ms" << std::endl;

		if (bFirst) {
			//t_timer.Start();
			g_CUDATSDFMerger->process(*g_CUDARGBDSensor, current_pose);
			//t_timer.Stop();
			//std::cout << "CUDATSDFMerger    process Time: " << t_timer.Elapsed() << " ms" << std::endl;
			bFirst = false;

		//	std::ofstream file("./outputs/model_0.ply");
		//	SaveData(file, g_CUDATSDFMerger->getModelData()->d_raycast_vertex, g_CUDATSDFMerger->getModelData()->d_raycast_normal, current_pose, w, h);
		}
		else {
			float4* frame_vertex = g_CUDARGBDSensor->getCUDASensorData()->d_vertex;
			float4* frame_normal = g_CUDARGBDSensor->getCUDASensorData()->d_normal;
			float4* model_vertex = g_CUDATSDFMerger->getModelData()->d_raycast_vertex;
			float4* model_normal = g_CUDATSDFMerger->getModelData()->d_raycast_normal;

			// Frame to Model transform (??) Is it model to frame motion?
			//t_timer.Start();
			float4x4 delta_transform = g_PointToPlaneICP->process(frame_vertex, frame_normal, model_vertex, model_normal);
			//t_timer.Stop();
			//std::cout << "PointToPlaneICP   process Time: " << t_timer.Elapsed() << " ms" << std::endl;
			if (delta_transform(0, 0) == -std::numeric_limits<float>::infinity()) {
				std::cout << "Fail ICP." << std::endl;
				continue;
			}

			// Set new transform.
			current_pose = current_pose * delta_transform;
			
			// TSDF update and raycast 
			//t_timer.Start();
			g_CUDATSDFMerger->process(*g_CUDARGBDSensor, current_pose);
			//t_timer.Stop();
			//std::cout << "CUDATSDFMerger    process Time: " << t_timer.Elapsed() << " ms" << std::endl;

		//	std::ofstream file("./outputs/model_0.ply");
		//	SaveData(file, model_vertex, model_normal, current_pose, w, h);

			// vertex, model
			cudaMemcpy(h_frame_normal, frame_normal, sizeof(float4)*w*h, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_model_normal, model_normal, sizeof(float4)*w*h, cudaMemcpyDeviceToHost);	// Updated model normal
			for (int v = 0; v < h; v++) {
				for (int u = 0; u < w; u++) 
				{
					cv::Vec4f pix, pix_blend;

					float alpha, beta = 0.5f;
					alpha = 1.0f - beta;

					float4 f4_f = h_frame_normal[v*w + u];
					float4 f4_m = h_model_normal[v*w + u];
					if (f4_f.x == MINF)
					{
						pix_blend.val[0] = f4_m.x;
						pix_blend.val[1] = f4_m.y;
						pix_blend.val[2] = f4_m.z;
						pix_blend.val[3] = f4_m.w;
					}
					else if (f4_m.x == MINF)
					{
						pix_blend.val[0] = f4_f.z;
						pix_blend.val[1] = f4_f.y;
						pix_blend.val[2] = f4_f.x;
						pix_blend.val[3] = f4_f.w;
					}
					else
					{
						pix_blend.val[0] = alpha*f4_f.z + beta*f4_m.x;
						pix_blend.val[1] = alpha*f4_f.y + beta*f4_m.y;
						pix_blend.val[2] = alpha*f4_f.x + beta*f4_m.z;
						pix_blend.val[3] = f4_f.w;
					}
					
					mBlendNormal.at<cv::Vec4f>(v * w + u) = pix_blend;

					pix.val[0] = f4_f.z;
					pix.val[1] = f4_f.y;
					pix.val[2] = f4_f.x;
					pix.val[3] = f4_f.w;
					mFrameNormal.at<cv::Vec4f>(v * w + u) = pix;

					pix.val[0] = f4_m.z;
					pix.val[1] = f4_m.y;
					pix.val[2] = f4_m.x;
					pix.val[3] = f4_m.w;
					mModelNormal.at<cv::Vec4f>(v * w + u) = pix;
				}
			}

			cv::imshow("frame normal", mFrameNormal);
			cv::imshow("model normal", mModelNormal);
			cv::imshow("blend normal", mBlendNormal);
			cv::waitKey(1);
		}
	}

	// Marching Cube
	g_CUDATSDFMarchingCube->process(*g_CUDATSDFMerger);


	///
	deleteAll();

	delete[] h_frame_normal;
	delete[] h_model_normal;

	return 0;
}
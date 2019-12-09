
#include "kf_main.h"

CUDARGBDSensor       *g_CUDARGBDSensor  = NULL;
CUDATSDFMerger       *g_CUDATSDFMerger  = NULL;
PointToPlaneICP      *g_PointToPlaneICP = NULL;

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
	volumeParam.voxel_origin = make_float3(VOXEL_ORIGIN_X, VOXEL_ORIGIN_Y, VOXEL_ORIGIN_Z);
	volumeParam.voxel_dim    = make_uint3(VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z);
	volumeParam.voxel_size   = VOXEL_SIZE;
	volumeParam.trunc_margin = TRUNC_MARGIN;

	// Create CUDATSDFMerger
	g_CUDATSDFMerger = new CUDATSDFMerger(volumeParam, sensorParam);
	g_CUDATSDFMerger->getVolumeData()->reset(volumeParam);

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

int main(int argc, char* argv[])
{
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
	int idx_start = 900;
	int frame_count = 600;

	float4x4 current_pose;
	current_pose.setIdentity();

	std::string data_path("../Dataset/Kinect1/ubody180/source");
	char depth_file[MAX_PATH] = {0,};
	char color_file[MAX_PATH] = {0,};

	for (int idx = idx_start; idx < idx_start + frame_count; idx++)
	{
		std::cout << "\n Frame idx: " << std::setw(10) << std::left << idx;
		memset(depth_file, 0x00, sizeof(char)*MAX_PATH);
		memset(color_file, 0x00, sizeof(char)*MAX_PATH);
		sprintf(depth_file, "%s/depth/d_%d.png", data_path.c_str(), idx);
		sprintf(color_file, "%s/color/c_%d.png", data_path.c_str(), idx);

		cv::Mat color = cv::imread(color_file);
		cv::Mat depth = cv::imread(depth_file, cv::IMREAD_ANYDEPTH);

		g_CUDARGBDSensor->process((uchar3*)color.data, (ushort*)depth.data);

		if (bFirst) {
			g_CUDATSDFMerger->process(*g_CUDARGBDSensor, &current_pose);
			bFirst = false;
		}
		else {
			float4* frame_vertex = g_CUDARGBDSensor->getCUDASensorData()->d_vertex;
			float4* frame_normal = g_CUDARGBDSensor->getCUDASensorData()->d_normal;
			float4* model_vertex = g_CUDATSDFMerger->getModelData()->d_raycast_vertex;
			float4* model_normal = g_CUDATSDFMerger->getModelData()->d_raycast_normal;

			// Frame to Model transform => Model to Frame transformation
			float4x4 delta_transform = g_PointToPlaneICP->process(frame_vertex, frame_normal, model_vertex, model_normal);
			if (delta_transform(0, 0) == -std::numeric_limits<float>::infinity()) {
				std::cout << "Fail ICP." << std::endl;
				continue;
			}

			// Set new transform.
			current_pose = current_pose * delta_transform;
			
			// TSDF update and raycast 
			g_CUDATSDFMerger->process(*g_CUDARGBDSensor, &current_pose);

			//SaveData(frame_vertex, model_vertex, transform, param.width, param.height);

			// vertex, model
			cudaMemcpy(h_frame_normal, frame_normal, sizeof(float4)*w*h, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_model_normal, model_normal, sizeof(float4)*w*h, cudaMemcpyDeviceToHost);
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

	///
	deleteAll();

	delete[] h_frame_normal;
	delete[] h_model_normal;

	return 0;
}
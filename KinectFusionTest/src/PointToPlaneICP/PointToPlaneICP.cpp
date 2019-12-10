
#include "PointToPlaneICP.h"

#include <opencv2/opencv.hpp>

#include "Eigen.h"

#define M_PI 3.14159265358979323846

//////////////////////////////////////////////////////////////////////////
//!
extern "C" void 
launch_compute_correspondences(PointToPlaneICP_Data&   icp_data, 
	                           PointToPlaneICP_Params& icp_params,
	                           float4x4&               delta_transform);
//////////////////////////////////////////////////////////////////////////


float4x4 PointToPlaneICP::process(float4* curr_frame_vertex, 
								  float4* curr_frame_normal, 
								  float4* prev_model_vertex, 
								  float4* prev_model_normal)
{
	float lastICPError = -1.0f;

	float4x4 transform_f2m;
	transform_f2m.setIdentity();

	size_t size = m_params.width*m_params.height;
	cudaMemcpy(m_data.d_frame_vertex, curr_frame_vertex, sizeof(float4)*size, cudaMemcpyDeviceToDevice); // [IN] for icp
	cudaMemcpy(m_data.d_frame_normal, curr_frame_normal, sizeof(float4)*size, cudaMemcpyDeviceToDevice); // [IN] for icp
	cudaMemcpy(m_data.d_model_vertex, prev_model_vertex, sizeof(float4)*size, cudaMemcpyDeviceToDevice); // [IN] for icp
	cudaMemcpy(m_data.d_model_normal, prev_model_normal, sizeof(float4)*size, cudaMemcpyDeviceToDevice); // [IN] for icp 

	// Level 

	for (int i = 0; i < m_params.max_outer_iter; i++)
	{
		// Establish a set of pair-correspondences.
		launch_compute_correspondences(m_data,
			                           m_params,
			                           transform_f2m);

		// Run ICP.
		LinearSystemConfidence confidence;
		transform_f2m = compute_icp(transform_f2m, confidence);
		if (std::abs(lastICPError - confidence.sumRegError) < m_params.early_out)
			break;

		lastICPError = confidence.sumRegError;
	}

	return transform_f2m;
}

float4x4 PointToPlaneICP::compute_icp(float4x4&               transform,
									  LinearSystemConfidence& confidence)
{
	float4x4 deltaTransform = transform;

	float4 mean = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float meanStDev = 1.0f;

	for (unsigned int i = 0; i < m_params.max_inner_iter; i++)
	{
		confidence.reset();
		Matrix6x7f system;

		m_linearSystem->apply(m_data.d_frame_vertex,
							  m_data.d_corre_vertex,
							  m_data.d_corre_normal,
							  deltaTransform,
							  m_params.width,
							  m_params.height,
							  system,
							  confidence);

		Matrix6x6f ATA = system.block(0, 0, 6, 6);
		Vector6f ATb = system.block(0, 6, 6, 1);

		if (ATA.isZero()) {
			return m_mtxICPLost;
		}

		Eigen::JacobiSVD<Matrix6x6f> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vector6f x = SVD.solve(ATb);

		//computing the matrix condition
		Vector6f evs = SVD.singularValues();
		confidence.matrixCondition = evs[0] / evs[5];

		float4x4 t = delinearize_transformation(x, Eigen::Vector3f(mean.x, mean.y, mean.z), meanStDev);
		if (t(0, 0) == -std::numeric_limits<float>::infinity()) {
			confidence.trackingLostThres = true;
			return m_mtxICPLost;
		}

		deltaTransform = t*deltaTransform;
	}

	return deltaTransform;
}

float4x4 PointToPlaneICP::delinearize_transformation(Vector6f& x, 
													 Eigen::Vector3f& mean, 
													 float meanStDev)
{
	Eigen::Matrix3f R = Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitZ()).toRotationMatrix()    // rotation Z (gamma)
		              * Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()    // rotation Y (beta)
		              * Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitX()).toRotationMatrix();   // rotation X (alpha)

	Eigen::Vector3f t = x.segment(3, 3);

	if (!check_rigid_transformation(R, t, m_params.thres_trans_angle, m_params.thres_trans_dist)) 
		return m_mtxICPLost;

	Eigen::Matrix4f res;
	res.setIdentity();
	res.block(0, 0, 3, 3) = R;
	res.block(0, 3, 3, 1) = meanStDev*t + mean - R*mean;

	float4x4 ret;
	ret.setIdentity();
	for (unsigned int i = 0; i < 16; i++) {
		ret.entries[i] = res.data()[i];
	}
	ret.transpose();

	return ret;
}

bool PointToPlaneICP::check_rigid_transformation(Eigen::Matrix3f& R, 
												 Eigen::Vector3f& t, 
												 float thres_trans_angle, 
												 float thres_trans_dist)
{
	Eigen::AngleAxisf aa(R);

	if (aa.angle() > thres_trans_angle || t.norm() > thres_trans_dist) {
		std::cout << "Tracking lost: angle " << (aa.angle() / M_PI)*180.0f << " translation " << t.norm() << std::endl;
		return false;
	}

	return true;
}
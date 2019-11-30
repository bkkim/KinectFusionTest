
#include "PointToPlaneICP.h"

#include <opencv2/opencv.hpp>

#include "Eigen.h"

#define M_PI 3.14159265358979323846

//////////////////////////////////////////////////////////////////////////
//!
extern "C" void 
launch_compute_correspondences(float4*   _frame_vertex, // [in] src
							   float4*   _frame_normal, // [in] src
							   float4*   _model_vertex, // [in] dst, target
							   float4*   _model_normal, // [in] dst, target
							   float4*   _corre_vertex, // [out]
							   float4*   _corre_normal, // [out] its type is float4* for using w value.
							   int       _width,
							   int       _height,
							   float     _fx,
							   float     _fy,
							   float     _cx,
							   float     _cy,
							   float     _z_min,
							   float     _z_max,
							   float     _thres_corres_dist,
							   float     _thres_corres_normal,
							   float4x4& _deltaTransform);
//////////////////////////////////////////////////////////////////////////


float4x4 PointToPlaneICP::process(float4* frame_vertex, 
								  float4* frame_normal, 
								  float4* model_vertex, 
								  float4* model_normal)
{
	float lastICPError = -1.0f;

	float4x4 transform;
	transform.setIdentity();

	for (int i = 0; i < m_params.max_outer_iter; i++)
	{
		// Establish a set of pair-correspondences.
		launch_compute_correspondences(frame_vertex, frame_normal, model_vertex, model_normal,
									   m_data.d_corres_vertex, m_data.d_corres_normal,
									   m_params.width, m_params.height,
									   m_params.fx, m_params.fy, m_params.cx, m_params.cy,
									   m_params.z_min, m_params.z_max,
									   m_params.thres_corres_dist, m_params.thres_corres_normal,
									   transform);


		// Run ICP.
		LinearSystemConfidence confidence;
		transform = compute_icp(frame_vertex, transform, confidence);
		if (std::abs(lastICPError - confidence.sumRegError) < m_params.early_out) {
		//	printf("numCorr     : %d\nsumRegError : %f\n", confidence.numCorr, confidence.sumRegError);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m11, transform.m12, transform.m13, transform.m14);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m21, transform.m22, transform.m23, transform.m24);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m31, transform.m32, transform.m33, transform.m34);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n\n", transform.m41, transform.m42, transform.m43, transform.m44);
			break;
		}
		else {
		//	printf("numCorr     : %d\nsumRegError : %f\n", confidence.numCorr, confidence.sumRegError);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m11, transform.m12, transform.m13, transform.m14);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m21, transform.m22, transform.m23, transform.m24);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m31, transform.m32, transform.m33, transform.m34);
		//	printf("%.8f\t%.8f\t%.8f\t%.8f\t\n", transform.m41, transform.m42, transform.m43, transform.m44);
		}

		lastICPError = confidence.sumRegError;
	}

	return transform;
}

float4x4 PointToPlaneICP::compute_icp(float4*                 frame_vertex,
									  float4x4&               transform,
									  LinearSystemConfidence& confidence)
{
	float4x4 deltaTransform = transform;
	deltaTransform.transpose();

	float4 mean = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float meanStDev = 1.0f;

	for (unsigned int i = 0; i < m_params.max_inner_iter; i++)
	{
		confidence.reset();
		Matrix6x7f system;

		m_linearSystem->apply(frame_vertex,
							  m_data.d_corres_vertex,
							  m_data.d_corres_normal,
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
		if (t(0, 0) == -std::numeric_limits<float>::infinity())
		{
			confidence.trackingLostThres = true;
			return m_mtxICPLost;
		}

		deltaTransform.transpose();
		deltaTransform = t*deltaTransform;
	}

	return deltaTransform;
}

float4x4 PointToPlaneICP::delinearize_transformation(Vector6f& x, 
													 Eigen::Vector3f& mean, 
													 float meanStDev)
{
	Eigen::Matrix3f R = Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()	// Rot Z
					  * Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()	// Rot Y
					  * Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitX()).toRotationMatrix();	// Rot X

	Eigen::Vector3f t = x.segment(3, 3);

	if (!check_rigid_transformation(R, t, m_params.thres_trans_angle, m_params.thres_trans_dist)) {
		return m_mtxICPLost;
	}

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
#pragma once

#include "cuda_SimpleMatrixUtil.h"

#include "PointToPlaneICP_Params.h"
#include "PointToPlaneICP_Data.h"

#include "LinearSystem.h"


class PointToPlaneICP
{
public:
	PointToPlaneICP(const PointToPlaneICP_Params& params)
	{
		m_params = params;
		create();
	}

	~PointToPlaneICP()
	{
		destroy();
	}

	float4x4 process(float4* curr_frame_vertex, 
		             float4* curr_frame_normal, 
		             float4* prev_model_vertex, 
		             float4* prev_model_normal);

private:
	void create() 
	{
		m_data.allocation(m_params);

		m_mtxICPLost.setValue(-std::numeric_limits<float>::infinity());
		m_linearSystem = new LinearSystem(m_params.width, m_params.height);
	}

	void destroy() {
		m_data.free();
	}

	float4x4 compute_icp(float4x4& transform,
						 LinearSystemConfidence& confidence);

	float4x4 delinearize_transformation(Vector6f& x);

	bool check_rigid_transformation(Eigen::Matrix3f& R, 
									Eigen::Vector3f& t, 
									float thres_trans_angle, 
									float thres_trans_dist);

public:
	bool m_bGPU;

	PointToPlaneICP_Params m_params;
	PointToPlaneICP_Data   m_data;

	float4x4               m_mtxICPLost;
	LinearSystem           *m_linearSystem;
};
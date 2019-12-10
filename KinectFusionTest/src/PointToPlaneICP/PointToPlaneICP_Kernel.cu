
#include "cuda_SimpleMatrixUtil.h"
#include "PointToPlaneICP_Params.h"
#include "PointToPlaneICP_Data.h"

#define T_PER_BLOCK		16

/////////////////////////////////////////////////////////////////////////
// For correspondence check

__global__ 
void compute_correspondences(float4*  _frame_vertex, 
	                         float4*  _frame_normal, 
	                         float4*  _model_vertex, 
	                         float4*  _model_normal,
	                         float4*  _corre_vertex, 
	                         float4*  _corre_normal, 
	                         int      _width, 
	                         int      _height,
	                         float    _fx, 
	                         float    _fy, 
	                         float    _cx, 
	                         float    _cy,
	                         float    _z_min, 
	                         float    _z_max, 
	                         float    _thres_corres_dist, 
	                         float    _thres_corres_normal,
	                         float4x4 _delta_transform)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= _width || y >= _height) return;

	_corre_vertex[y*_width + x] = make_float4(MINF, MINF, MINF, MINF);
	_corre_normal[y*_width + x] = make_float4(MINF, MINF, MINF, MINF);

	float4 pFv = _frame_vertex[y*_width + x];
	float4 pFn = _frame_normal[y*_width + x];
	float4 pFc = make_float4(MINF, MINF, MINF, MINF);

	if (pFv.x != MINF && pFn.x != MINF)
	{
		float4 pTv = _delta_transform * pFv; // it is a vertex
		float4 pTn = _delta_transform * pFn; // it is a normal
	
		uint u = (uint)(((pTv.x * _fx) / pTv.z) + _cx);
		uint v = (uint)(((pTv.y * _fy) / pTv.z) + _cy);
	
		if (u >= 0 && u < _width && v >= 0 && v < _height)
		{
			unsigned int idx = v * _width + u;
			float4 pMv = _model_vertex[idx];
			float4 pMn = _model_normal[idx];
	
			if (pMv.x != MINF && pMn.x != MINF)
			{
				float dist_vertex = length(pTv - pMv);
				float dist_normal = dot(pTn, pMn);
	
				if (dist_vertex <= _thres_corres_dist && dist_normal >= _thres_corres_normal)
				{
					float w = max(0.0, 0.5f*((1.0f - dist_vertex / _thres_corres_dist) + (1.0f - ((pTv.z - _z_min) / (_z_max - _z_min)))));	// for weighted ICP
					
					_corre_vertex[y*_width + x] = pMv;
					_corre_normal[y*_width + x] = make_float4(pMn.x, pMn.y, pMn.z, w);
				}
			}
		}
	}
}

extern "C" void
launch_compute_correspondences(PointToPlaneICP_Data&   icp_data,
	                           PointToPlaneICP_Params& icp_params,
	                           float4x4&               delta_transform)
{
	const dim3 gridSize((icp_params.width + T_PER_BLOCK - 1) / T_PER_BLOCK, (icp_params.height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	compute_correspondences <<< gridSize, blockSize >>> (icp_data.d_frame_vertex,
		                                                 icp_data.d_frame_normal,
		                                                 icp_data.d_model_vertex,
		                                                 icp_data.d_model_normal,
		                                                 icp_data.d_corre_vertex,
		                                                 icp_data.d_corre_normal,
		                                                 icp_params.width,
		                                                 icp_params.height,
		                                                 icp_params.fx,
		                                                 icp_params.fy,
		                                                 icp_params.cx,
		                                                 icp_params.cy,
		                                                 icp_params.z_min,
		                                                 icp_params.z_max,
		                                                 icp_params.thres_corres_dist,
		                                                 icp_params.thres_corres_normal,
		                                                 delta_transform);
#ifdef _DEBUG
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError(__FUNCTION__);
#endif

}

#pragma once

struct PointToPlaneICP_Params
{
	int	  width;
	int   height;
	
	float fx;
	float fy;
	float cx;
	float cy;

	float z_min;
	float z_max;

	int   level;
	int   max_inner_iter;
	int   max_outer_iter;
	float thres_corres_dist;
	float thres_corres_normal;
	float thres_trans_angle;
	float thres_trans_dist;
	float thres_cond;
	float early_out;
};
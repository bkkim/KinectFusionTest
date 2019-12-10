#pragma once

//#include <iostream>
//#include <fstream>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "cuda_SimpleMatrixUtil.h"
#include "CUDARGBDSensor.h"
#include "CUDATSDFMerger.h"
#include "PointToPlaneICP.h"
#include "kf_util.h"

#include <opencv2/opencv.hpp>

#define INPUT_WIDTH				640
#define INPUT_HEIGHT			480
#define DEPTH_MIN				0
#define DEPTH_MAX				6
#define KINECT_FOCAL_X			525
#define KINECT_FOCAL_Y			525
#define KINECT_CENTER_X			320
#define KINECT_CENTER_Y			240

#define FILTER_SPATIAL			0.5
#define FILTER_RANGE			1.0
#define FILTER_OBJECT_MIN		0.7
#define FILTER_OBJECT_MAX		1.5

#define VOLUME_ORIGIN_X			-0.512//-1.024
#define VOLUME_ORIGIN_Y			-0.512//-1.024
#define VOLUME_ORIGIN_Z			0.7

#define VOLUME_DIM_X			256
#define VOLUME_DIM_Y			256
#define VOLUME_DIM_Z			256

#define VOXEL_SIZE				0.004
#define TRUNC_MARGIN			VOXEL_SIZE * 10

#define ICP_LEVEL				1
#define ICP_MAX_INNER_ITER		1
#define ICP_MAX_OUTER_ITER		8
#define ICP_THRES_CORRES_DIST	0.08f//0.15f
#define ICP_THRES_CORRES_NORMAL	0.1f //0.97f
#define ICP_THRES_TRANS_ANGLE	0.4f //1.0f
#define ICP_THRES_TRANS_DIST	0.1f //1.0f
#define ICP_THRES_COND			100.0f
#define ICP_EARLY_OUT			0.01f
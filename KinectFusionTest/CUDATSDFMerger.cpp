
#include "CUDATSDFMerger.h"
#include "SensorUtil.h"

CUDATSDFMerger::CUDATSDFMerger(VolumeParam volumeParam, SensorParam sensorParam)
	: m_volumeData(NULL)
{
#ifdef _DEBUG
	std::cout << "volume dimensions ( " << volumeParam.voxel_dim.x << " x " << volumeParam.voxel_dim.y << " x " << volumeParam.voxel_dim.z << " )" << std::endl;
#endif

	m_volumeParam = volumeParam;
	m_sensorParam = sensorParam;
	m_volumeData  = new VolumeData(volumeParam);
	m_modelData   = new ModelData(sensorParam);
}

CUDATSDFMerger::~CUDATSDFMerger()
{
	SAFE_DELETE(m_volumeData);
	SAFE_DELETE(m_modelData);
}

void CUDATSDFMerger::process(CUDARGBDSensor& sensor, float4x4* transform)
{
	CUDASensorData *cudaSensorData = sensor.getCUDASensorData();
	
	// Update TSDF
	tsdf::cuda::update(
		m_sensorParam.intrinsic, 
		*transform,
		m_sensorParam.width, 
		m_sensorParam.height,
		cudaSensorData->d_depth_foreground,
		cudaSensorData->d_color,
		m_volumeParam.voxel_dim, 
		m_volumeParam.voxel_origin, 
		m_volumeParam.voxel_size, 
		m_volumeParam.trunc_margin,
		m_volumeData->d_tsdf, 
		m_volumeData->d_tsdf_weight, 
		m_volumeData->d_color, 
		m_volumeData->d_color_weight);

	////////////////////////////////////
	// Make ModelData from TSDF Volume.
	tsdf::cuda::depth_raycast(
		m_sensorParam.intrinsic, 
		*transform,
		m_sensorParam.width, 
		m_sensorParam.height,
		m_volumeParam.voxel_origin, 
		m_volumeParam.voxel_dim, 
		m_volumeParam.voxel_size, 
		m_volumeParam.trunc_margin,
		m_volumeData->d_tsdf, 
		m_volumeData->d_tsdf_weight,
		m_modelData->d_raycast_depth);
	
	util::cuda::convert_depth2vertex(
		m_modelData->d_raycast_vertex, 
		m_modelData->d_raycast_depth, 
		m_sensorParam.intrinsic.getInverse(), 
		m_sensorParam.width, 
		m_sensorParam.height);
	
	util::cuda::compute_normals(
		m_modelData->d_raycast_normal, 
		m_modelData->d_raycast_vertex, 
		m_sensorParam.width, 
		m_sensorParam.height);
	////////////////////////////////////

	return;
}
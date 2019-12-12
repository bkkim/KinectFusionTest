
#include "CUDATSDFMerger.h"
#include "SensorUtil.h"
#include "gpu_timer.h"

CUDATSDFMerger::CUDATSDFMerger(VolumeParam volumeParam, SensorParam sensorParam)
	: m_volumeData(NULL)
{
#ifdef _DEBUG
	std::cout << "volume dimensions ( " << volumeParam.volume_dim.x << " x " << volumeParam.volume_dim.y << " x " << volumeParam.volume_dim.z << " )" << std::endl;
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

void CUDATSDFMerger::process(CUDARGBDSensor& sensor, float4x4& current_pose)
{
	CUDASensorData *cudaSensorData = sensor.getCUDASensorData();
	
	GpuTimer t_timer;

	// Update TSDF
	t_timer.Start();
	tsdf::cuda::update(
		m_sensorParam.intrinsic, 
		current_pose.getInverse(),
		m_sensorParam.width, 
		m_sensorParam.height,
		cudaSensorData->d_depth_foreground,
		cudaSensorData->d_color,
		m_volumeParam.volume_dim, 
		m_volumeParam.volume_origin, 
		m_volumeParam.voxel_size, 
		m_volumeParam.trunc_margin,
		m_volumeData->d_tsdf, 
		m_volumeData->d_tsdf_weight, 
		m_volumeData->d_color, 
		m_volumeData->d_color_weight);
	t_timer.Stop();
	std::cout << " - CUDATSDFMerger  update Time: " << t_timer.Elapsed() << " ms" << std::endl;

	////////////////////////////////////
	// Make ModelData from TSDF Volume.
	t_timer.Start();
	tsdf::cuda::raycast(
		m_sensorParam.intrinsic, 
		current_pose,
		m_sensorParam.width, 
		m_sensorParam.height,
		m_volumeParam.volume_origin, 
		m_volumeParam.volume_dim, 
		m_volumeParam.voxel_size, 
		m_volumeParam.trunc_margin,
		m_volumeData->d_tsdf, 
		m_volumeData->d_tsdf_weight,
		m_modelData->d_raycast_depth,
		m_modelData->d_raycast_vertex,
		m_modelData->d_raycast_normal,
		m_modelData->d_raycast_color);
	t_timer.Stop();
	std::cout << " - CUDATSDFMerger raycast Time: " << t_timer.Elapsed() << " ms" << std::endl;

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

#include <cuda_runtime.h>

#include "CUDARGBDSensor.h"
#include "SensorUtil.h"

CUDARGBDSensor::CUDARGBDSensor(RGBDSensor* sensor)
{
	m_sensor = sensor;
	m_data = new CUDASensorData(sensor->getSensorParam());
}

CUDARGBDSensor::~CUDARGBDSensor()
{
	SAFE_DELETE(m_data);
}

HRESULT CUDARGBDSensor::process(uchar3* color, ushort* depth)
{
	HRESULT hr = S_OK;
	if (m_sensor->process(color, depth) == S_FALSE)
		return S_FALSE;
	
	SensorData  *sensorData = m_sensor->getSensorData();
	SensorParam sensorParam = m_sensor->getSensorParam();
	
	// memory copy host to device
	checkCudaErrors(cudaMemcpy(m_data->d_depth, sensorData->m_depth, sizeof(float )*sensorParam.width*sensorParam.height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_data->d_color, sensorData->m_color, sizeof(uchar3)*sensorParam.width*sensorParam.height, cudaMemcpyHostToDevice));

	util::cuda::bilateral_filter(    m_data->d_depth_filtered, 
								     m_data->d_depth, 
								     m_filterParam.sigma_d, 
								     m_filterParam.sigma_r, 
								     sensorParam.width, 
								     sensorParam.height);
	util::cuda::subtract_foreground( m_data->d_depth_foreground, 
									 m_data->d_depth_filtered, 
									 m_filterParam.foreground_depth_min, 
									 m_filterParam.foreground_depth_max, 
									 sensorParam.width, 
									 sensorParam.height);
	util::cuda::convert_depth2vertex(m_data->d_vertex, 
									 m_data->d_depth_foreground, 
									 sensorParam.intrinsic.getInverse(), 
									 sensorParam.width, 
									 sensorParam.height);
	util::cuda::compute_normals(     m_data->d_normal, 
								     m_data->d_vertex, 
								     sensorParam.width, 
								     sensorParam.height);

	return hr;
}
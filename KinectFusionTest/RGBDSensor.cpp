
#include "RGBDSensor.h"

RGBDSensor::RGBDSensor(SensorParam param)
	: m_data(NULL)
{
#ifdef _DEBUG
	std::cout << "sensor dimensions ( " << param.width << ", " << param.height << " )" << std::endl;
#endif
	m_sensorParam = param;
	m_data = new SensorData(param);
}

RGBDSensor::~RGBDSensor()
{
	SAFE_DELETE(m_data);
}


HRESULT RGBDSensor::process(uchar3* color, ushort* depth)
{
	HRESULT hr = S_OK;

	for (int i = 0; i < m_sensorParam.width * m_sensorParam.height; i++)
	{
		if (depth[i] == 0)
			m_data->m_depth[i] = -std::numeric_limits<float>::infinity();
		else
			m_data->m_depth[i] = (float)depth[i] * 0.001;
	}

	for (int i = 0; i < m_sensorParam.width * m_sensorParam.height; i++)
	{
		m_data->m_color[i] = color[i];
	}

	return hr;
}


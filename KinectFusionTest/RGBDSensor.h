#pragma once

#include <windows.h>
#include <cassert>

#include "SensorData.h"

class RGBDSensor
{
public:
	RGBDSensor(SensorParam param);
	~RGBDSensor();

	HRESULT process(uchar3* color, ushort* depth);

	SensorParam getSensorParam()
	{
		return m_sensorParam;
	}

	SensorData*	getSensorData()
	{
		return m_data;
	}

protected:

	SensorParam m_sensorParam;
	SensorData	*m_data;
};
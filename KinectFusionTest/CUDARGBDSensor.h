#pragma once

#include "RGBDSensor.h"
#include "FilterParam.h"
#include "CUDASensorData.h"

class CUDARGBDSensor
{
public:
	CUDARGBDSensor(RGBDSensor* sensor);
	~CUDARGBDSensor();

	HRESULT process(uchar3* color, ushort* depth);

	RGBDSensor* getRGBDSensor() {
		return m_sensor;
	}
	
	CUDASensorData* getCUDASensorData() {
		return m_data;
	}

	void setFilterParam(FilterParam param) {
		m_filterParam = param;
	}

	FilterParam getFilterParam() {
		return m_filterParam;
	}

private:

	FilterParam    m_filterParam;
	RGBDSensor     *m_sensor;
	CUDASensorData *m_data;
};
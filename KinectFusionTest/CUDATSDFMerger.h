#pragma once

#include "VolumeData.h"
#include "ModelData.h"
#include "CUDARGBDSensor.h"
#include "TSDFUpdate.h"
#include "TSDFRaycast.h"

class CUDATSDFMerger
{
public:
	CUDATSDFMerger(VolumeParam volumeParam, SensorParam sensorParam);
	~CUDATSDFMerger();

	void process(CUDARGBDSensor& sensor, float4x4* transform=NULL);

	VolumeParam getVolumeParam() {
		return m_volumeParam;
	}

	VolumeData* getVolumeData() {
		return m_volumeData;
	}

	ModelData* getModelData() {
		return m_modelData;
	}

private:
	VolumeParam	m_volumeParam;
	SensorParam m_sensorParam;
	VolumeData  *m_volumeData;
	ModelData   *m_modelData;
};
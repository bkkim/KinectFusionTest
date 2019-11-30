#pragma once

#include "SensorParam.h"

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
#endif

class SensorData
{
public:
	SensorData(SensorParam param)
		: m_depth(NULL)
		, m_color(NULL)
	{
		create(param);
	}

	~SensorData()
	{
		destroy();
	}

private:
	void create(SensorParam param) 
	{
		SAFE_DELETE(m_depth);
		m_depth = new float[param.width*param.height];

		SAFE_DELETE(m_color);
		m_color = new uchar3[param.width*param.height];
	}

	void destroy()
	{
		SAFE_DELETE(m_depth);
		SAFE_DELETE(m_color);
	}

public:

	float		*m_depth;
	uchar3		*m_color;
};
#pragma once

/************************************************************************/
/* Might be used to determine whether camera tracking is lost or not    */
/************************************************************************/

#include <iostream>
#include <vector>
#include <list>
#include <cassert>
#include <fstream>

struct LinearSystemConfidence {
	LinearSystemConfidence() {
		reset();
	}

	void reset() {
		sumRegError = 0.0f;
		sumRegWeight = 0.0f;
		numCorr = 0;
		matrixCondition = 0.0f;
		trackingLostThres = false;
	}

	void print() const {
		std::cout <<
			"relRegError \t" << sumRegError / sumRegWeight << "\n" <<
			"sumRegError:\t" << sumRegError << "\n" <<
			"sumRegWeight:\t" << sumRegWeight << "\n" <<
			"numCorrespon:\t" << numCorr << "\n" <<
			"matrixCondit:\t" << matrixCondition << "\n\n";
	}

	bool isTrackingLost() const {
		const float			threshMatrixCondition = 150.0f;
		const float			threshSumRegError = 2000.0f;

		const float			threshRelError = 1.5f;

		if (sumRegError > threshSumRegError)				return true;
		if (matrixCondition > threshMatrixCondition)		return true;
		if (sumRegError / sumRegWeight > threshRelError)	return true;
		if (trackingLostThres)								return true;

		return false;
	}

	float sumRegError;
	float sumRegWeight;
	unsigned int numCorr;
	float matrixCondition;
	bool trackingLostThres;
};
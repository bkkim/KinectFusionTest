/*
* This code is a modified version of the marching-cube algorithm supported by NVIDIA
* Modified by bkkim79@keti.re.kr
*/

/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Marching cubes

This sample extracts a geometric isosurface from a volume dataset using
the marching cubes algorithm. It uses the scan (prefix sum) function from
the Thrust library to perform stream compaction.  Similar techniques can
be used for other problems that require a variable-sized output per
thread.

For more information on marching cubes see:
http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
http://en.wikipedia.org/wiki/Marching_cubes

Volume data courtesy:
http://www9.informatik.uni-erlangen.de/External/vollib/

For more information on the Thrust library1
http://code.google.com/p/thrust/

The algorithm consists of several stages:

1. Execute "classifyVoxel" kernel
This evaluates the volume at the corners of each voxel and computes the
number of vertices each voxel will generate.
It is executed using one thread per voxel.
It writes two arrays - voxelOccupied and voxelVertices to global memory.
voxelOccupied is a flag indicating if the voxel is non-empty.

2. Scan "voxelOccupied" array (using Thrust scan)
Read back the total number of occupied voxels from GPU to CPU.
This is the sum of the last value of the exclusive scan and the last
input value.

3. Execute "compactVoxels" kernel
This compacts the voxelOccupied array to get rid of empty voxels.
This allows us to run the complex "generateTriangles" kernel on only
the occupied voxels.

4. Scan voxelVertices array
This gives the start address for the vertex data for each voxel.
We read back the total number of vertices generated from GPU to CPU.

Note that by using a custom scan function we could combine the above two
scan operations above into a single operation.

5. Execute "generateTriangles" kernel
This runs only on the occupied voxels.
It looks up the field values again and generates the triangle data,
using the results of the scan to write the output to the correct addresses.
The marching cubes look-up tables are stored in 1D textures.

6. Render geometry
Using number of vertices from readback.
*/
#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <helper_cuda.h>

#include "cuda_SimpleMatrixUtil.h"

#include "defines.h"

#include "CUDATSDFMerger.h"


//////////////////////////////////////////////////////////////////////////
//!
extern "C" void
launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
float3 voxelSize, float isoValue);

extern "C" void
launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied,
uint *voxelOccupiedScan, uint numVoxels);

extern "C" void
launch_generateTriangles(dim3 grid, dim3 threads,
float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void
launch_generateTriangles2(dim3 grid, dim3 threads,
float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 gridOrigins,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);
extern "C" void bindVolumeTexture(uchar *d_volume);
extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements);


class CUDATSDFMarchingCube
{
public:
	CUDATSDFMarchingCube();
	~CUDATSDFMarchingCube();

public:
	HRESULT create(uint3 gridSize, float3 origins, float voxelSize, float isoValue);
	HRESULT destroy();

	uint process(CUDATSDFMerger& merger);

	void setVolume(float *volume);

	float4 *getPos() {
		return d_pos;
	}

	float4 *getNormal() {
		return d_normal;
	}

	uchar3 *getColor() {
		return d_color;
	}

	uint getMaxVerts() {
		return m_maxVerts;
	}

	uint getTotalVerts() {
		return m_totalVerts;
	}

private:

	uint3 m_gridSizeLog2;
	uint3 m_gridSizeShift;
	uint3 m_gridSize;
	uint3 m_gridSizeMask;
	float3 m_gridOrigins;

	float3 m_voxelSize;
	uint m_numVoxels;
	uint m_maxVerts;
	uint m_activeVoxels;
	uint m_totalVerts;

	float m_isoValue;
	float m_dIsoValue;

	// device data
	float4 *d_pos;
	float4* d_normal;
	uchar3* d_color;

	float *d_volume;
	uint *d_voxelVerts;
	uint *d_voxelVertsScan;
	uint *d_voxelOccupied;
	uint *d_voxelOccupiedScan;
	uint *d_compVoxelArray;

	// tables
	uint *d_numVertsTable;
	uint *d_edgeTable;
	uint *d_triTable;
};
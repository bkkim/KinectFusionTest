
#include "CUDATSDFMarchingCube.h"

CUDATSDFMarchingCube::CUDATSDFMarchingCube()
: m_gridSizeLog2()
, m_gridSizeShift()
, m_gridSize()
, m_gridSizeMask()
, m_gridOrigins()
, m_voxelSize()
, m_numVoxels(0)
, m_maxVerts(0)
, m_activeVoxels(0)
, m_totalVerts(0)
, m_isoValue(0.0f)
, m_dIsoValue(0.005f)
{

}

CUDATSDFMarchingCube::~CUDATSDFMarchingCube()
{
	
}

HRESULT CUDATSDFMarchingCube::create(uint3 gridSize, float3 gridOrigins, float voxelSize, float isoValue)
{
	HRESULT hr = S_OK;

	m_gridSizeLog2.x = log2(gridSize.x);
	m_gridSizeLog2.y = log2(gridSize.y);
	m_gridSizeLog2.z = log2(gridSize.z);

	m_gridSize = make_uint3(1 << m_gridSizeLog2.x, 1 << m_gridSizeLog2.y, 1 << m_gridSizeLog2.z);
	m_gridSizeMask = make_uint3(m_gridSize.x - 1, m_gridSize.y - 1, m_gridSize.z - 1);
	m_gridSizeShift = make_uint3(0, m_gridSizeLog2.x, m_gridSizeLog2.x + m_gridSizeLog2.y);
	m_gridOrigins = gridOrigins;

	m_numVoxels = m_gridSize.x * m_gridSize.y * m_gridSize.z;
	m_voxelSize = make_float3(voxelSize, voxelSize, voxelSize);
	m_maxVerts = m_gridSize.x * m_gridSize.y * 100;
	m_isoValue = isoValue;

#ifdef _DEBUG
	printf("grid: %d x %d x %d = %d voxels\n", m_gridSize.x, m_gridSize.y, m_gridSize.z, m_numVoxels);
	printf("max verts = %d\n", m_maxVerts);
#endif

 	checkCudaErrors(cudaMalloc((void **)&(d_pos), m_maxVerts * sizeof(float)* 4));
	checkCudaErrors(cudaMalloc((void **)&(d_normal), m_maxVerts * sizeof(float)* 4));
	checkCudaErrors(cudaMalloc((void **)&(d_color), m_maxVerts * sizeof(uchar) * 3));
	// allocate texture
	allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

	// allocate volume data
//	int size = m_gridSize.x * m_gridSize.y * m_gridSize.z * sizeof(float);
//	checkCudaErrors(cudaMalloc((void **)&d_volume, size));

	// allocate device memory
	unsigned int memSize = sizeof(uint)* m_numVoxels;
	checkCudaErrors(cudaMalloc((void **)&d_voxelVerts, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelVertsScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupied, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupiedScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_compVoxelArray, memSize));

	return hr;
}

HRESULT CUDATSDFMarchingCube::destroy()
{
	HRESULT hr = S_OK;

	checkCudaErrors(cudaFree(d_pos));
	checkCudaErrors(cudaFree(d_normal));
	checkCudaErrors(cudaFree(d_color));

	checkCudaErrors(cudaFree(d_edgeTable));
	checkCudaErrors(cudaFree(d_triTable));
	checkCudaErrors(cudaFree(d_numVertsTable));

	checkCudaErrors(cudaFree(d_voxelVerts));
	checkCudaErrors(cudaFree(d_voxelVertsScan));
	checkCudaErrors(cudaFree(d_voxelOccupied));
	checkCudaErrors(cudaFree(d_voxelOccupiedScan));
	checkCudaErrors(cudaFree(d_compVoxelArray));

	return hr;
}

uint CUDATSDFMarchingCube::process(CUDATSDFMerger& merger)
{
	int threads = 128;
	dim3 grid(m_numVoxels / threads, 1, 1);

	float* d_volume = merger.getVolumeData()->d_tsdf;
	float isoValue = m_isoValue;

	// get around maximum grid size of 65535 in each dimension
	if (grid.x > 65535)
	{
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	// calculate number of vertices need per voxel
	launch_classifyVoxel(grid, threads,
		d_voxelVerts, d_voxelOccupied, d_volume,
		m_gridSize, m_gridSizeShift, m_gridSizeMask,
		m_numVoxels, m_voxelSize, isoValue);

#if SKIP_EMPTY_VOXELS
	// scan voxel occupied array
	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, m_numVoxels);

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelOccupied + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelOccupiedScan + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		m_activeVoxels = lastElement + lastScanElement;
	}

	if (m_activeVoxels == 0)
	{
		// return if there are no full voxels
		m_totalVerts = 0;
		return 0;
	}

	// compact voxel index array
	launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, m_numVoxels);
	getLastCudaError("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS

	// scan voxel vertex count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, m_numVoxels);

	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelVerts + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelVertsScan + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		m_totalVerts = lastElement + lastScanElement;
	}

#if SKIP_EMPTY_VOXELS
	dim3 grid2((int)ceil(m_activeVoxels / (float)NTHREADS), 1, 1);
#else
	dim3 grid2((int)ceil(m_numVoxels / (float)NTHREADS), 1, 1);
#endif

	while (grid2.x > 65535)
	{
		grid2.x /= 2;
		grid2.y *= 2;
	}

#if SAMPLE_VOLUME
	launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal,
		d_compVoxelArray,
		d_voxelVertsScan, d_volume,
		m_gridSize, m_gridSizeShift, m_gridSizeMask, m_gridOrigins,
		m_voxelSize, isoValue, m_activeVoxels,
		m_maxVerts);
#else
	launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal,
		d_compVoxelArray,
		d_voxelVertsScan,
		m_gridSize, m_gridSizeShift, m_gridSizeMask,
		m_voxelSize, isoValue, m_activeVoxels,
		m_maxVerts);
#endif

	return m_totalVerts;
}

void CUDATSDFMarchingCube::setVolume(float *volume)
{
	int size = m_gridSize.x*m_gridSize.y*m_gridSize.z*sizeof(float);
	checkCudaErrors(cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice));
	//bindVolumeTexture(d_volume);
}


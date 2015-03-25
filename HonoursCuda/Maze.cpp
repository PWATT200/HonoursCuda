#include "Maze.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#include <chrono>



Maze::Maze(unsigned width, unsigned height) : m_width(width), m_height(height)
{
	m_maze = new bool[m_width * m_height];
}

__device__ void Maze::Generate(){
	Initialize();
	Carve(2, 2);
	m_maze[m_width + 2] = true;
	m_maze[(m_height - 2) * m_width + m_width - 3] = true;
}


__device__ void Maze::Initialize(){
	int length = m_width * m_height;

	for (int i = 0; i < length; ++i)
	{
		m_maze[i] = false;
	}

	for (unsigned x = 0; x < m_width; x++) {
		m_maze[x] = true;
		m_maze[(m_height - 1) * m_width + x] = true;
	}
	for (unsigned y = 0; y < m_height; y++) {
		m_maze[y * m_width] = true;
		m_maze[y * m_width + m_width - 1] = true;
	}

}


__device__ void Maze::Carve(int x, int y){

	m_maze[y * m_width + x] = true;

	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);


	const unsigned d = curand_uniform_double(&state);


	for (unsigned i = 0; i < 4; i++) {
		const int dirs[] = { 1, -1, 0, 0 };
		const int dx = dirs[(i + d + 0) % 4];
		const int dy = dirs[(i + d + 2) % 4];
		const int x1 = x + dx, y1 = y + dy;
		const int x2 = x1 + dx, y2 = y1 + dy;
		if (!m_maze[y1 * m_width + x1] && !m_maze[y2 * m_width + x2]) {
			m_maze[y1 * m_width + x1] = true;
			Carve(x2, y2);
		}
	}
}

std::ostream &Maze::Show(std::ostream &os) const{
	for (unsigned y = 0; y < m_height; y++) {
		for (unsigned x = 0; x < m_width; x++) {
			os << (m_maze[y * m_width + x] ? "  " : "[]");
		}
		os << "\n";
	}
	return os;
}

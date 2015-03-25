#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <iostream>

class Maze
{
public:

	const unsigned m_width;
	const unsigned m_height;
	bool * m_maze = new bool[];
	int randomNum;

	friend std::ostream &operator<<(std::ostream &os, const Maze &maze);

	Maze();

	CUDA_CALLABLE_MEMBER Maze(unsigned width, unsigned height);
	CUDA_CALLABLE_MEMBER void Generate();
	CUDA_CALLABLE_MEMBER void Initialize();
	CUDA_CALLABLE_MEMBER void Carve(int x, int y);
	std::ostream &Show(std::ostream &os) const;

};
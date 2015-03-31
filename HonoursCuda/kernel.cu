
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>

#define N 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

class Maze
{

	const unsigned m_width = 39;
	const unsigned m_height = 23;
	bool m_maze[897];
	curandState_t mState;
	
public:
	
	friend std::ostream &operator<<(std::ostream &os, const Maze &maze);
	/*


	__device__ Maze(unsigned width, unsigned height);
	__device__ void Generate();
	__device__ void Initialize();
	__device__ void Carve(int x, int y);
	std::ostream &Show(std::ostream &os) const;*/

	Maze()
	{
	}

	__device__ void setState(curandState_t state){
		mState = state;
	}

	__device__ void Generate(){
		Initialize();
		Carve(2, 2);
		m_maze[m_width + 2] = true;
		m_maze[(m_height - 2) * m_width + m_width - 3] = true;
	}


	__device__ void Initialize(){
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



	__device__ void Carve(int x, int y){
		

		m_maze[y * m_width + x] = true;
		const unsigned d = curand(&mState) % 100;;

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
			//__syncthreads();
		}
		
}

	std::ostream &Show(std::ostream &os) const{
		for (unsigned y = 0; y < m_height; y++) {
			for (unsigned x = 0; x < m_width; x++) {
				os << (m_maze[y * m_width + x] ? "  " : "[]");
			}
			os << "\n";
		}
		return os;
	}


};





__global__ void createMaze(Maze * mArray, curandState_t *state)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < N) {

		//*intArray = curand(&state[idx]) % 100;
		mArray[idx].setState(state[idx]);
		mArray[idx].Generate();
	
	}
}


/** Maze insertion operator. */
std::ostream &operator<<(std::ostream &os, const Maze &maze)
{
	return maze.Show(os);
}





__global__ void setupRandStates(curandState_t* state, unsigned int seed) {
	unsigned block_id = blockIdx.y * gridDim.x + blockIdx.x;
	int thread_id = threadIdx.x + block_id * blockDim.x;
	// Each thread gets same seed, a different sequence number, no offset
	curand_init(seed, thread_id, 0, &state[thread_id]);
	
}




int main(int argc, char *argv[])
{
	//const int N = 50;

	

	// Allocate space on the device to store the random states

	curandState_t h_randStates[N];
	curandState_t* d_randStates;

	//set stack size for the recursive function depth
	size_t myStackSize = N*sizeof(Maze);
	cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);
	

	/* allocate space on the GPU for the random states */
	gpuErrchk(cudaMalloc((void**)&d_randStates, N * sizeof(curandState_t)));
	

	//cudaMemcpy(d_randStates, &h_randStates, N*sizeof(int), cudaMemcpyHostToDevice);
	

	cudaEvent_t rng_start, rng_stop;
	checkCuda(cudaEventCreate(&rng_start));
	checkCuda(cudaEventCreate(&rng_stop));

	cudaEventRecord(rng_start, 0);
	// Setup the randStates
	setupRandStates <<<N, 1 >>>(d_randStates, time(NULL));
	cudaEventRecord(rng_stop, 0);

	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaGetLastError());
	
	//gpuErrchk(cudaMemcpy(h_randStates, d_randStates, N*sizeof(curandState_t), cudaMemcpyDeviceToHost));
	
	
	std::chrono::time_point<std::chrono::system_clock> start, end, startTest, endTest;
	
	Maze h_mArray[N];
	Maze *d_MArray;

	cudaEvent_t gpu_start, gpu_stop;
	checkCuda(cudaEventCreate(&gpu_start));
	checkCuda(cudaEventCreate(&gpu_stop));
	
	

	gpuErrchk(cudaMalloc((void**)&d_MArray, N* sizeof(Maze)));
	gpuErrchk(cudaMemcpy(d_MArray, &h_mArray, N* sizeof(Maze) , cudaMemcpyHostToDevice));

	start = std::chrono::system_clock::now();

	cudaEventRecord(gpu_start, 0);
	//Generate the Mazes
	createMaze<<<N, 1 >>>(d_MArray, d_randStates);
	cudaEventRecord(gpu_stop, 0);

	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaGetLastError());


	//gpuErrchk(cudaMemcpy(h_randStates, d_randStates, N*sizeof(int), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(h_mArray, d_MArray, N* sizeof(Maze), cudaMemcpyDeviceToHost));


	
	end = std::chrono::system_clock::now();

	
	//std::cout << intArray << " - ";
	for (int i = 0; i < 100; ++i){
		std::cout << h_mArray[i];
	}


	
	std::chrono::duration<double> interval = end - start;
	std::cout << "Cuda Test No 1 - iteration : 1000 elasped time: " << interval.count() << "s\n";

	cudaFree(d_MArray);
	cudaFree(d_randStates);

	cudaDeviceReset();
	return 0;
}

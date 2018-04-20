/* Adapted from Robert Hochberg, created almost entirely from the CUDA C programming guide*/
// Concluded on 20th April, 2018
//This hopes to be the C++ version
// This code will do parallel MCMC

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cstdio>
#include <cstdlib> // for min/max
#include <curand_kernel.h> // random number

#define BLOCK_SIZE 2 // size or dimension of the matrix per se
using namespace std;

// error return code

//debug outputs
#define CUDA_KERNEL_DEBUG 0 //test for illegal memory access
#define OUTPUT_PRE 1 // preprocess debug output
#define OUTPUT_POST 1 //postprocess debyg output


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		std::cout << "GPUassert: " << cudaGetErrorString(code) << " / " << file << " " << line << std::endl;
		//fcout << stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
	else {
		if (CUDA_KERNEL_DEBUG == 1) {
			std::cout << "success GPUassert: " << cudaGetErrorString(code) << " / " << file << " " << line << std::endl;
		}
	}
}

// Header
// Matrices are stored in row major order
//M(row, col) = *(M.elements + row * M.width + col)

//declaring matrix structure

typedef struct {
	int width;
	int height;
	float* elements;
}Matrix;

// DEVICE CODE

// matrix multiplication kernel prototype
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// random number kernels 
//seed
__device__ void init(float seed, curandState_t* states) { // 'states' is the state of the system in time 

	int iid = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, iid, 0, &states[iid]);
}
//random number kernel
__device__ float rndn(float* no, curandState_t* states) { // you can change the data type to unsigned int to give +ve numbers only

	int iid = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t localState = states[iid]; // copy to local memory
	for (int n = 0; n < BLOCK_SIZE; n++) {
		//float rno = curand_uniform(&localState);     
		return no[iid] = curand_uniform(&localState);
	}
	states[iid] = localState; // to refresh the generator state
	return no[iid];
}

// Matrix multiplication kernel function definition for device
// device function to check matrix size
__device__ void checkmatrixsize(Matrix A, Matrix B, Matrix C) {

	if (A.width != B.height)

		printf("Your matrix dimension cannot be multiplied, Please check your inputs and try again");

	else if (A.width == B.height)

		printf(" Your matrix dimension match and can be multiplied, you are good to go! ");
		
}
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	
	curandState_t* seedgen;// for the seed and the rando number generation
	float seed; // for the initi function
	float* d_rn; // for the random numbers
	curandState* rngen;
	
	init(seed,  seedgen);
	rndn(d_rn, rngen); // try using same curandState* in seed for rndn



	float Cvalue = 0.0;
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > A.height || col > B.width) return;
	//or
	if (row < BLOCK_SIZE && col < BLOCK_SIZE)
		//or
	checkmatrixsize;// this calls the device function defined above
		
	for (int e = 0; e < A.width; ++e) {
		(A.elements[row * A.width + e]) = d_rn[A.width * A.height]; // putting the random bumber in A.elements
		(B.elements[e * B.width + col]) = d_rn[B.width * B.height]; // putting the random number in B.elements
			Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
		}
	__syncthreads();
	C.elements[row * C.width + col] = Cvalue;
}


// HOST CODE

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
	
	// uncomment below if you dont want to use the curand device api for the multiplication thus you must also adjust the client side
	/*
	// seed random number
	curandState_t* rn;
	float h_rn;
	float* d_rn;

	gpuErrchk(cudaMalloc((void**)&rn, BLOCK_SIZE * sizeof(curandState_t)));
	gpuErrchk(cudaMalloc((void**)&d_rn, BLOCK_SIZE * sizeof(float)));
	
	init << < dimGrid, dimBlock >> > (time(0), rn); // seed
//invoke kernel to launch the random numbers 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
		(A.height + dimBlock.y - 1) / dimBlock.y);
	rndn << <  dimGrid, dimBlock >> > (d_rn, rn);

	//gpuErrchk(cudaMemcpy(h_rn, d_rn, N * sizeof(float), cudaMemcpyDeviceToHost));
	*/
	// ALLOCATE & COPY A, B & C to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * BLOCK_SIZE * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_A.elements, size));
	//copy A to device. declare A.elements in client side
	gpuErrchk(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * BLOCK_SIZE * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_B.elements, size));
	//copy B to device. declare B.elements in clients side
	gpuErrchk(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_C.elements, size));

	//invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
		(A.height + dimBlock.y - 1) / dimBlock.y);
	MatMulKernel <<< dimGrid, dimBlock >> > (d_A, d_B, d_C);
	gpuErrchk(cudaThreadSynchronize());
	// Read C from device memory
	gpuErrchk(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));
	// free memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	//cudaFree(d_rn);
}

//CLIENT SIDE

int main(int argc, char* argv[]) {
	// generate random numbers to pour into matrix elements

	//curandGenerator_t gen;
	//curandSetPseudoRandomGeneratorSeed (gen, 1234ULL);
	Matrix T, S, R; //T = transition, S= status, R = resultant
	// get the sizes of the matrix

	int tH, tW, sH, sW, rH, rW; // width and height of the T & S matrices
	tH = BLOCK_SIZE; /* Height of transtion matrix */
	tW = BLOCK_SIZE; /* Width of transtion matrix */
	sH = tW; /* Height of status vector */
	sW = 1; // check this!! it is the width of the status vector
	rH = tH;
	rW = 0;
	for (int w = 0; w < BLOCK_SIZE; w++)
	{
		rW += tW * sH;
	}


	T.height = tH;
	T.width = tW;
	S.height = sH;
	S.width = sW;
	R.width = rW;
	R.height = rH;

	//elements

	T.elements = new float[T.width * T.height * sizeof(float)];
	S.elements = new float[S.width * S.height * sizeof(float)];
	R.elements = new float[R.width * R.height * sizeof(float)];

	// initialize the contents of the elements, this tells how they are aranged

	for (int i = 0; i < T.height; i++) {
		for (int j = 0; j < T.width; j++) {

			T.elements[i * T.width + j]; // add the random numbers here if error occurs
		}
	}
	for (int i = 0; i < S.height; i++) {
		for (int j = 0; j < S.width; j++) {
			S.elements[i * S.width + j]; // add the random numbers here if error occurs
		}
	}
	// call the Host function

	MatMul(T, S, R);


	// print the 5 x 5 portion of the T, S and R matrices
	for (int i = 0; i < __min(5, T.height); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, T.width); j++) {
			cout << " Transtion elements are: \n" << endl;
			cout << T.elements[i * T.width + j] << endl;
		}
	}
	cout << "\n" << endl;

	for (int i = 0; i < __min(5, S.height); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, S.width); j++) {
			cout << " Status vector elements are: \n" << endl;
			cout << S.elements[i * S.width + j] << endl;
		}
	}
	cout << "\n" << endl;

	for (int i = 0; i < __min(5, R.height); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, R.width); j++) {
			cout << " Resultant elements are: \n" << endl;
			cout << R.elements[i * R.width + j] << endl;
		}
	}
	cout << "\n" << endl;
	
}
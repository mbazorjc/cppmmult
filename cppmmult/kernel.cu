/* Adapted from Robert Hochberg, created almost entirely from the CUDA C programming guide*/
// https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
// Concluded on 20th April, 2018
//This hopes to be the C++ version of mostly the client side
// This code will do parallel MCMC

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <ctime>
#include <random> // for c++11 random number generation on host
#include <cstdio>
#include <cstdlib> // for min/max
#include <curand_kernel.h> // random number
#include <device_functions.h>

#define BLOCK_SIZE 256 // size or dimension of the matrix per se
#define NLOOP 1000000 // size or dimension of the matrix per se
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

// DEVICE CODE

// random number kernels 
//seed is now called in the rndn, uncomment below to call seed separately
//__global__ void init(float seed, curandState_t* states) { // 'states' is the state of the system in time 
//
//	int iid = blockDim.x * blockIdx.x + threadIdx.x;
//
//	curand_init(seed, iid, 0, &states[iid]);
//}

//random number kernel for device function
__device__ float* rndn (float* nrdn,  curandState_t* states) { // pointer returning function ^___^

	int iid = blockDim.x * blockIdx.x + threadIdx.x;
	
		curand_init(clock(), iid, 0, &states[iid]); //same seed for each thread, to avoid passing seed from outside
		curandState_t localState = states[iid]; // copy to local memory
		nrdn[iid] = curand_uniform(&localState);
	 // generates random number
		states[iid] = localState;
	// to refresh the generator state
	return nrdn;
}
//__global__ void setrnkernels(float* _ptr, curandState* globalState, const unsigned int _points)
//{
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//	//only call gen on the kernels we have inited
//	//(one per device container element)
//	if (idx < _points)
//	{
//		_ptr[idx] = rndno(globalState, idx);
//	}
//}



//__global__ void init(float seed, curandState_t* states) { // 'states' is the state of the system in time 
//
//	int iid = blockDim.x * blockIdx.x + threadIdx.x;
//	curand_init(seed, iid, 0, &states[iid]);
//}
//random number kernel
//__global__ void rndn (float* no, curandState_t* states) { // you can change the data type to unsigned int to give +ve numbers only
//
//	int iid = blockDim.x * blockIdx.x + threadIdx.x;
//	//curand_init((float)clock(), iid, 0, &states[iid]); //seed
//	curandState_t localState = states[iid]; // copy to local memory
//	for (int n = 0; n < BLOCK_SIZE; n++) {
//		//float rno = curand_uniform(&localState);     
//		no[iid] = curand_uniform(&localState);
//	}
//	states[iid] = localState; // to refresh the generator state
//	
//}
// Matrix multiplication kernel function definition for device
// device function to check matrix size
//__device__ void checkmatrixsize(Matrix A, Matrix B, Matrix C) {
//
//	if (A.width != B.height)
//
//		printf("Your matrix dimension cannot be multiplied, Please check your inputs and try again");
//
//	else if (A.width == B.height)
//
//		printf(" Your matrix dimension match and can be multiplied, you are good to go! ");
//		
//}
__global__ void MatMulKernel( float* d_A, const int lda,  float* d_B, float* d_C) {
	

	//lda = leading dimension of A
	
	// Each thread computes one element of C
		
	//thread id
	int eleidx = threadIdx.x;
	int blockidx = blockIdx.x;
	int blockdimx = blockDim.x;
	int blockidy = blockIdx.y;

	//shared memory
	extern __shared__ float temp[]; // temporary array for reduction
	
	float Cvalue = 0.0;
	curandState_t* rn;
	//shared memory (reduction method) for matrix A and B
	d_A = rndn(d_A, rn);
	d_B = rndn(d_B, rn ); // putting the random numbers in d_B

	temp[eleidx] = d_A[blockidy * lda + blockidx * blockdimx + eleidx] * d_B[blockidx * blockdimx + eleidx];

	__syncthreads();

	//do reduction in shared memory (adapted from udacity class, lecture 3 code snippet)
	for (unsigned int i = blockDim.x / 2; i > 0; i = i/2) { // this reduces threadblock by half untill the calculation is done
		
		if (eleidx < i) {
			temp[eleidx] = temp[eleidx] + temp[eleidx + i];
		}
		__syncthreads();
		
	}

	// atomically write result back to global memory using thread 0
	if (eleidx == 0){
		//float input = temp[0];
		//float rslt = d_C[blockidy];
		atomicAdd( &d_C[blockidy], temp[0]);
	}
	Cvalue = d_C[blockidy];
	// print out d_C

	printf("the values of C are: ", Cvalue);



}


// HOST CODE

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
//MatMul(Tele, Twidth, Theight, Sele, Swidth, Sheight, Rele, Rwidth, Rheight); //kernel

void MatMul(const float* A, const int Aw, const int Ah, const float* B, const int Bw, const int Bh, float* C, const int Cw, const int Ch) {
	//0. common variable
	dim3 dimBlock, dimGrid;

	//1. allocate device memory
	// uncomment below and also the cudaMemcpyHosttoDevice if you dont want to use the curand device api for the multiplication thus you must also adjust the client side
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
	size_t size;
	float *d_A,*d_B,*d_C; //Transition matrix (initial state is given)
	
	size = Aw * Ah * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_A, size));
	gpuErrchk(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	size = Bw * Bh * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_B, size));
	//gpuErrchk(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

	size = Cw * Ch * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_C, size));

		
	// the probability of being a state is obtained by determining the probability of being a state and minus 1
		

	/*dimBlock.x = BLOCK_SIZE; dimBlock.y = 1; dimBlock.z = 1;
	dimGrid.x = (Bw / dimBlock.x) + 1; dimGrid.y = Bh; dimGrid.z = 1;
	init <<< dimGrid, dimBlock >> > (d_B, rngen); */

	// declare a state, a seed, a generator, create Bh random number inside d_B;



	cout << "finished allocating device memory and copying, now launching the MatMultKernel on device. . . \n\n";


	//3. invoke kernel
	dimBlock.x = BLOCK_SIZE; dimBlock.y = 1; dimBlock.z = 1;
	dimGrid.x = (Aw/ dimBlock.x)+1; dimGrid.y = Ah; dimGrid.z = 1; //Ah = Bh = Ch, Aw, Bw = Cw = 1
	for (int i = 0; i < NLOOP; i++) {
		MatMulKernel <<< dimGrid, 1000, dimBlock.x >>> (d_A, Aw, d_B, d_C);
		gpuErrchk(cudaMemcpy(d_B, d_C, size, cudaMemcpyDeviceToDevice));// this should be repeated for the nmatix number of times
	}

	//4. Read C from device memory
	gpuErrchk(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
	
	//5. Cleanup
	// free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	//cudaFree(rngen);
}

//CLIENT SIDE

int main(int argc, char* argv[]) {
	
	//parameters
	//int nstatus = 8192; //16384 is limit for GT650M
	//float deltat = (float)1 / 12; // 1 week
	//float pf = (float)0.01; // 1 year
	//float totaltime = 5; // 10 years
	//int nsim = 16384;
	//int nsteps = (int)(totaltime / deltat);

	//1. PREPROCESSING
	//float *Tele, *Sele, *Rele; //T = transition, S= status, R = resultant
	int tH, tW, sH, sW, rH, rW; // width and height of the T & S matrices
						
	
	tH = 10; /* Height of transtion matrix */
	tW = 10; /* Width of transtion matrix */
	int Theight = tH;
	int Twidth = tW;
	float* Tele = (float*)malloc(Twidth * Theight * sizeof(float)); // memory on heap

	sH = tW; /* Height of status vector */
	sW = 10; // check this!! it is the width of the status vector
	int Sheight = sH;
	int Swidth = 10;
	float* Sele = (float*)malloc(Swidth * Sheight * sizeof(float));
	
	rH = tH;
	rW = tW;
	int Rwidth = rW;
	int Rheight = rH;
	float* Rele = (float*)malloc(Rwidth * Rheight * sizeof(float));

	//elements
	cout << "allocating memory in host to contain the device's transfer . . . \n";
	// generate host random number to fill the T, S and R elements created with "new"
	//float seed = time(0);
	//mt19937_64 h_rn(seed); // or any engine u choose like "random_device"
	//	

	// initialize the contents of the elements with the random numbers, this tells how they are aranged

	for (int i = 0; i < Theight; i++) {
		for (int j = 0; j < Twidth; j++) {

			Tele[i * Twidth + j] = rand();// = h_rn(); // add the random numbers here if error occurs
		}
	}
	
	//2. WORKHORSE
	// call the Host function

	MatMul(Tele, Twidth, Theight, Sele, Swidth, Sheight, Rele, Rwidth, Rheight); //kernel

	//3. POSTPROCESSING
	// print the 5 x 5 portion of the T, S and R matrices
	for (int i = 0; i < __min(5, Theight); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, Twidth); j++) {
			
			cout << " Transtion elements are: \n" << Tele[i * Twidth + j] << endl;
		}
	}
	cout << "\n" << endl;
	

	for (int i = 0; i < __min(5, Sheight); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, Swidth); j++) {
			
			cout << " Status vector elements are: \n" << Sele[i * Swidth + j] << endl;
		}
	}
	cout << "\n" << endl;

	for (int i = 0; i < __min(5, Rheight); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, Rwidth); j++) {
			
			cout << " Resultant elements are: \n" << Rele[i * Rwidth + j] << endl;
		}
	}
	cout << "\n" << endl;

	// free memory
	free(Tele);
	
	free(Sele);
	
	free(Rele);
	
	cin.get();

	return 0;
}
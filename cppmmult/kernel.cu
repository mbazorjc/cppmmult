/* Adapted from Robert Hochberg, created almost entirely from the CUDA C programming guide*/
// https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
// Concluded on 20th April, 2018
//This hopes to be the C++ version of mostly the client side
// This code will do parallel MCMC
// https://www.youtube.com/watch?v=aZhaYO_cV6I // for main arguements
// to copy the transtion txt file from another directroy to this project output directory


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
#include <time.h>
#include <fstream>
#include <string>

#define BLOCK_SIZE 256 // size or dimension of the matrix per se

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
__device__ float rndn (float* nrdn,  curandState_t* states) { // pointer returning function ^___^

	int iid = blockDim.x * blockIdx.x + threadIdx.x;
	
		curand_init(clock(), iid, 0, &states[iid]); //same seed for each thread, to avoid passing seed from outside
		curandState_t localState = states[iid]; // copy to local memory
		nrdn[iid] = curand_uniform(&localState);
	 // generates random number
		states[iid] = localState;
	// to refresh the generator state
	return nrdn[iid];
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
__global__ void gMatMulKernel( const float* d_A, const int lda,  float* d_B, float* d_C) {
	

	//lda = leading dimension of A
	
	// Each thread computes one element of C
		
	//thread id
	unsigned int eleidx = threadIdx.x;
	unsigned int blockidx = blockIdx.x;
	unsigned int blockdimx = blockDim.x;
	unsigned int blockidy = blockIdx.y;

	//shared memory
	extern __shared__ float temp_sh[]; // temporary array for reduction
	
	//curandState_t* rn;
	//shared memory (reduction method) for matrix A and B
	//d_A = rndn(d_A, rn);
	//d_B = rndn(d_B, rn ); // putting the random numbers in d_B
	if ( (blockidx * blockdimx + eleidx) < lda) {
		temp_sh[eleidx] = d_A[blockidy * lda + blockidx * blockdimx + eleidx] * d_B[blockidx * blockdimx + eleidx];
	}
	__syncthreads();

	//do reduction in shared memory (adapted from udacity class, lecture 3 code snippet)
	for (unsigned int i = blockDim.x / 2; i > 0; i = i/2) { // this reduces threadblock by half untill the calculation is done
		
		if (eleidx < i) {
			temp_sh[eleidx] = temp_sh[eleidx] + temp_sh[eleidx + i];
		}
		__syncthreads();
		
	}

	// atomically write result back to global memory using thread 0
	if (eleidx == 0){
		//float input = temp[0];
		//float rslt = d_C[blockidy];
		atomicAdd( &d_C[blockidy], temp_sh[0]);
	}
	// print out d_C
	//float Cvalue = d_C[blockidy];
	//printf("the values of C are: ", Cvalue);
	}

//set initial B
//	gSetInitialVector <<< dimGrid, dimBlock, dimBlock.x >>> (d_B, Bh, value);
__global__ void gSetInitialVector(float* d_B, const int Bh, const float value)
{
	//thread id
	int eleidx = threadIdx.x;
	int blockidx = blockIdx.x;
	int blockdimx = blockDim.x;

	//set the value of B
	if (blockidx * blockdimx + eleidx < Bh) {
		d_B[blockidx * blockdimx + eleidx] = value;
	}

	//done

}

// HOST CODE

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
// MatMul(tele, tw, th, sele, sh, rele, rh, nloop); //kernel
void MatMul(const float* Aele, const int Aw, const int Ah, float* Bele, const int Bh, float* Cele, const int Ch, const unsigned int nloop) {
	//0. common variable
	dim3 dimBlock, dimGrid;

	//1. allocate device memory

	// ALLOCATE & COPY A, B & C to device memory
	size_t size;
	float *d_A,*d_B,*d_C; //Transition matrix (initial state is given)
	
	//transition matrix
	gpuErrchk(cudaMalloc((void**)&d_A, sizeof(float) * Aw * Ah ));
	

	//input state (state t-1)
	size = Bh * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_B, size));

	//output vector (state t)
	size = Ch * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_C, size));

	//2. copy transition matrix and initialize the state vector at t = 0 to random number
	// to be loaded from file
	size = Aw * Ah * sizeof(float);
	gpuErrchk(cudaMemcpy(d_A, Aele, size, cudaMemcpyHostToDevice));
	// load transtion matrix from file


	//workaround for starting position
	size = Bh * sizeof(float);
	const float initb = 0.5;
	dimBlock.x = BLOCK_SIZE; dimBlock.y = 1; dimBlock.z = 1;
	dimGrid.x = (Bh / dimBlock.x) + 1; dimGrid.y = 1; dimGrid.z = 1;
	gSetInitialVector <<< dimGrid, dimBlock >>> (d_B, Bh, initb);

	gpuErrchk( cudaMemcpy(Bele, d_B, size, cudaMemcpyDeviceToHost) );

	// random number allocation for B (to be fix later)
	/*curandState_t* rn;
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
	//gpuErrchk(cudaMemcpy(h_rn, d_rn, N * sizeof(float), cudaMemcpyDeviceToHost));*/
	cout << "finished allocating device memory and copying, now launching the MatMultKernel on device. . . \n\n";


	//3. invoke kernel
	dimBlock.x = BLOCK_SIZE; dimBlock.y = 1; dimBlock.z = 1;
	dimGrid.x = (Aw / dimBlock.x)+1; dimGrid.y = Ah; dimGrid.z = 1; //Ah = Bh = Ch, Aw, Bw = Cw = 1

	cout << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << endl;
	cout << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << endl;


	for (int i = 0; i < nloop/2; i++) {
		//cout << "," << i;
		gMatMulKernel <<< dimGrid, dimBlock, ( dimBlock.x *sizeof(float) ) >>> (d_A, Aw, d_B, d_C);
		gMatMulKernel <<< dimGrid, dimBlock,( dimBlock.x * sizeof(float) ) >>> (d_A, Aw, d_C, d_B);

		//gpuErrchk(cudaMemcpy(d_B, d_C, sizeof(float)*Bh, cudaMemcpyDeviceToDevice));// this should be repeated for the nmatix number of times
	}
	cout << endl;

	//4. Read C from device memory
	gpuErrchk( cudaMemcpy(Cele, d_B, sizeof(float)*Ch , cudaMemcpyDeviceToHost));
	
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
	
	//float *Tele, *Sele, *Rele; //T = transition, S= status, R = resultant
	const unsigned int nloop = 1000; // number of loop
	const unsigned int th = 3000; //number of row
	const unsigned int tw = th; //number of column
	const unsigned int sh = th; //number of row
	const unsigned int rh = th; //number of row

	//1. PREPROCESSING
	float *tele, *sele, *rele;
	gpuErrchk(cudaHostAlloc((void**)&tele, sizeof(float)*th * tw, cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&sele, sizeof(float)*sh * 1, cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&rele, sizeof(float)*rh * 1, cudaHostAllocDefault));

	//elements
	cout << "allocating memory in host. . . \n";
	// generate host random number to fill the T, S and R elements created with "new"
	//float seed = time(0);
	//mt19937_64 h_rn(seed); // or any engine u choose like "random_device"
	//	

	// initialize the contents of the elements with the random numbers, this tells how they are aranged
	// later the transition matrix must be loaded from a file based on the system blueprint.
	random_device gen;
	default_random_engine(time(0));
	uniform_real_distribution<float> rn(0.0, 1.0);
	
	//test
	if (argc != 2) {
		cout << "not enough arguements\n";
		cin.get();
		return 1;
	}

	cout << "now taking the transtion matrix parameters from file" << endl;
	//for (int i = 0; i < th; i++) {
	//	for (int j = 0; j < tw; j++) {
	//		tele[i * tw + j] = rn(gen); // add the random numbers here if error occurs

			ifstream infile(argv[1]); // from the txt file

			//test
			if (!infile.good()) {
				cout << "file read error on: " << argv[1] << endl;
				cin.get();
				return 1;
			}
			// read line by line the file
			while (!infile.eof()) {
				string data;
				getline(infile, data);
				cout << data << endl;
			}
			cout << endl;
	
	

	// ok
	
	//2. WORKHORSE
	// call the Host function

	MatMul(tele, tw, th, sele, sh, rele, rh, nloop); //kernel

	//3. POSTPROCESSING
	// print the 5 x 5 portion of the T, S and R matrices
	cout << " Transition elements are: ";
	for (int i = 0; i < __min(5, th); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		for (int j = 0; j < __min(5, tw); j++) {
			cout << tele[i * tw + j] << " , ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << " Status vector elements are: ";
	for (int i = 0; i < __min(5, sh); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
		 cout << sele[i] << " , ";
	}
	cout << endl;

	cout << " Resultant elements are: ";
	for (int i = 0; i < __min(5, rh); i++) { //__min() is used because min() did not work, the former is a c++ macro while the later is a c macro 
			cout << rele[i] << " , ";
	}
	cout << endl;

	// free memory
	gpuErrchk( cudaFreeHost(tele) );
	gpuErrchk( cudaFreeHost(sele) );
	gpuErrchk( cudaFreeHost(rele) );
	cin.get();

	return 0;
}
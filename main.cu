#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>
#include <chrono>

// A,Gの行列のサイズ N x N
const int N = 100;

// バッチサイズ
const int C = 5000;

// 計算回数
const int CALC = 100;

// A kp B * vec(X)
void k(float *A,float *B,float *X,float *R,cublasHandle_t cublas){
	float one = 1.0f,zero = 0.0f;
	cublasSgemm(
			cublas,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			N,N,N,
			&one,
			A,N,
			X,N,
			&zero,
			R,N
			);
	cublasSgemm(
			cublas,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			N,N,N,
			&one,
			R,N,
			B,N,
			&zero,
			R,N
			);
}

// バッチで
void batched_k(float *A[C],float *B[C],float *X[C],float *R[C],cublasHandle_t cublas[C]){
	for(int i = 0;i < C;i++){
		k(A[i],B[i],X[i],R[i],cublas[i]);
	}
}

int main(){
	float *A[C];
	float *B[C];
	float *X[C];
	float *R[C];
	cublasHandle_t cublas[C];
	cudaStream_t stream[C];

	std::cout<<"行列サイズ : "<<N<<" x "<<N<<std::endl
		<<"バッチサイズ : "<<C<<std::endl
		<<"計算回数 : "<<CALC<<std::endl;

	// 初期化
	for(int i = 0;i < C;i++){
		cudaMalloc((void**)&A[i],sizeof(float)*N*N);
		cudaMalloc((void**)&B[i],sizeof(float)*N*N);
		cudaMalloc((void**)&X[i],sizeof(float)*N*N);
		cudaMalloc((void**)&R[i],sizeof(float)*N*N);
		// cublasの用意とstreamの接続
		cudaStreamCreate(stream+i);
		cublasCreate( cublas+i );
		cublasSetStream( cublas[i], stream[i]);
	}
	// ウォームアップ
	batched_k(A,B,X,R,cublas);
#ifdef BATCHED
		auto start = std::chrono::system_clock::now();
		for(int i = 0;i < CALC;i++)batched_k(A,B,X,R,cublas);
		cudaDeviceSynchronize();
		auto stop = std::chrono::system_clock::now();
		std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()/static_cast<float>(CALC)<<"[ms]"<<std::endl;
#else	
		cublasHandle_t cub;
		cublasCreate( &cub );
		auto start = std::chrono::system_clock::now();
		for(int j = 0;j < CALC;j++){
			for(int i = 0;i < C;i++){
				k(A[i],B[i],X[i],R[i],cub);
			}
		}
		cudaDeviceSynchronize();
		auto stop = std::chrono::system_clock::now();
		std::cout<<"seq "<<std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()/static_cast<float>(CALC)<<"[ms]"<<std::endl;
		cublasDestroy( cub );
#endif

	for(int i = 0;i < C;i++){
		cudaFree(A[i]);
		cudaFree(B[i]);
		cudaFree(X[i]);
		cudaFree(R[i]);
		cublasDestroy( cublas[i] );
		cudaStreamDestroy( stream[i] );
	}

}

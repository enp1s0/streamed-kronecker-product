#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>
#include <chrono>

// A,Gの行列のサイズ N x N (デフォルト値)
const int DEF_N = 200;

// バッチサイズ
const int C = 30;

// 計算回数
const int CALC = 10;

// A kp B * vec(X)
void k(float *A,float *B,float *X,float *R,int N,cublasHandle_t cublas){
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
	// Rがちゃんと計算されているかは知らない
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
void batched_k(float *A[C],float *B[C],float *X[C],float *R[C],int N,cublasHandle_t cublas[C]){
	for(int i = 0;i < C;i++){
		k(A[i],B[i],X[i],R[i],N,cublas[i]);
	}
}

int main(int argc,char **argv){
	float *A[C];
	float *B[C];
	float *X[C];
	float *R[C];
	cublasHandle_t cublas[C];
	cudaStream_t stream[C];

	int N = DEF_N;

	if(argc > 1){
		N = std::stoi(argv[1]);
	}


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
	batched_k(A,B,X,R,N,cublas);
#ifdef BATCHED
		auto start = std::chrono::system_clock::now();
		for(int i = 0;i < CALC;i++)batched_k(A,B,X,R,N,cublas);
		cudaDeviceSynchronize();
		auto stop = std::chrono::system_clock::now();
		std::cout<<"計算時間 : "<<std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()/static_cast<float>(CALC)<<" [ms]"<<std::endl;
#else	
		cublasHandle_t cub;
		cublasCreate( &cub );
		auto start = std::chrono::system_clock::now();
		for(int j = 0;j < CALC;j++){
			for(int i = 0;i < C;i++){
				k(A[i],B[i],X[i],R[i],N,cub);
			}
		}
		cudaDeviceSynchronize();
		auto stop = std::chrono::system_clock::now();
		std::cout<<"計算時間 : "<<std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()/static_cast<float>(CALC)<<" [ms]"<<std::endl;
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

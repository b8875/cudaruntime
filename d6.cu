#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BLOCKS 12
#define BLOCKSIZE 1024
//#define BSize 32
//#define QSize (BLOCKS*BLOCKSIZE)/BSize/32
#define BSize 12
#define QSize 32

struct kernel_para{
volatile int *A, *B, *C;
volatile int size;
volatile int block;
volatile int thread;
volatile int warp;
volatile int req;
volatile int funcId;
volatile int taskId;
volatile int doneHost;
int doneGPU;
};

struct kernel_para_GPU{
int warpId;
int baseId;
int queueId;
int locId;
int taskId;
};

typedef struct {
int contents[BSize][QSize]; // body of queue
//int last[BSize]; // position of last element
}queue;

__device__ void init_queue(queue *q, int *warp, int *j){
        int tid = blockDim.x*blockIdx.x+threadIdx.x;
	warp[tid/32/QSize] = 0;
	j[tid/32/QSize] = 0;
	if((tid % QSize) == 0){
		q->contents[tid/32/QSize][0] = 1;
	}else{
		q->contents[tid/32/QSize][tid%QSize] = 0;
	}
}

__device__ void MatMul_kernel(int *A, int *B, int *C, int M_height, int M_width, int N_width, int baseTid){
#if 1
        int row = baseTid + (threadIdx.x & 0x1f);
        if(row < M_height) {
                for (int j = 0; j < N_width; j++){
                        int sum = 0;
                        for (int k = 0; k < M_width; k++){
                                int a = A[row * M_width + k];
                                int b = B[k * N_width + j];
                                sum += a * b;

                        }
                        C[row * N_width + j] = sum;

                }
        }
#endif
}

__global__ void deviceRT(volatile int *done, volatile int *totalExecTasks, volatile kernel_para_GPU *warpPool, volatile struct kernel_para *taskBuffer, struct kernel_para *taskArgs, queue *warpQ){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int warpIdxx = (blockIdx.x*blockDim.x + threadIdx.x)/32;
	__shared__ int warp[BSize];
	__shared__ int j[BSize];
	int threadDone;
	if ((warpIdxx%QSize) == 0){
		init_queue(warpQ, warp, j);
	}
	__syncthreads();
	if((warpIdxx%QSize) == 0){
		threadDone = 0;
//		printf("Scheduling:%d, %d\n", warpIdxx, warpIdxx/16);
		while(!(*done)){
			if(*done) continue;
			if((tid%32) < QSize){
				if(taskBuffer[warpIdxx/QSize].req == 1 && !(*done)){
					warp[warpIdxx/QSize] = taskBuffer[warpIdxx/QSize].warp;
//					if((threadIdx.x & 0x1f) == 0) printf("Scheduling:%d, %d\n", taskBuffer[warpIdxx/QSize].taskId, warpIdxx/QSize);
					while(1){
						threadDone = 0;
						if(warpQ->contents[warpIdxx/QSize][tid%QSize] == 0){
//						threadDone = 0;
							if(atomicSub(&warp[warpIdxx/QSize],1) > 0){
					//			printf("Scheduling:%d, %d\n", taskBuffer[warpIdxx/QSize].taskId, warpIdxx);
								warpPool[warpIdxx+(tid%QSize)].queueId = warpIdxx/QSize;
                                                        	warpPool[warpIdxx+(tid%QSize)].locId = tid%QSize;
								warpQ->contents[warpIdxx/QSize][tid%QSize] = 1;
								warpPool[warpIdxx+(tid%QSize)].baseId = atomicAdd(&j[warpIdxx/QSize],1)*32;
								warpPool[warpIdxx+(tid%QSize)].taskId = taskBuffer[warpIdxx/QSize].taskId;
								warpPool[warpIdxx+(tid%QSize)].warpId = 1;
//								__threadfence();
								__threadfence_block();
							}
						}
						if(warp[warpIdxx/QSize] < 0){
							threadDone = 1;
						}
						if(__all(threadDone == 1) == 1){
		//	                        while(__all(threadDone == 1) == 0);
							taskBuffer[warpIdxx/QSize].req = 0;
							j[warpIdxx/QSize] = 0;
							break;
						}
#if 0
						warpQ->last[tid]++;
                                                if(warpQ->last[tid] == QSize){
                                                        if(tid == 0){
                                                                warpQ->last[tid] = 1;
                                                        }else{
                                                                warpQ->last[tid] = 0;
                                                        }
                                                }
#endif
			
					
					}
//					threadDone=1;
//                                        while(__all(threadDone == 1) == 0 );
//					if((threadIdx.x & 0x1f) == 0) printf("Scheduling:%d\n", taskBuffer[warpIdxx/QSize].taskId);
//					taskBuffer[warpIdxx/QSize].req = 0;
//					j[warpIdxx/QSize] = 0;
				}
			}
		}
	}else{
#if 1
		int warpIdx = (blockIdx.x*blockDim.x + threadIdx.x)/32;
		while(!(*done)){
			while(warpPool[warpIdx].warpId == 0 && !(*done));
			if(*done) return;
			switch(taskArgs[warpPool[warpIdx].taskId].funcId){
				case 1:
//				if((threadIdx.x & 0x1f) == 0) printf("Before:%d, %d\n", warpPool[warpIdx].taskId, warpPool[warpIdx].baseId);
				MatMul_kernel((int*)taskArgs[warpPool[warpIdx].taskId].A, (int*)taskArgs[warpPool[warpIdx].taskId].B, (int*)taskArgs[warpPool[warpIdx].taskId].C, taskArgs[warpPool[warpIdx].taskId].size, taskArgs[warpPool[warpIdx].taskId].size, taskArgs[warpPool[warpIdx].taskId].size, warpPool[warpIdx].baseId);
//                                      if((threadIdx.x & 0x1f) == 0) printf("After:%d, %d\n", warpPool[warpIdx].taskId, warpPool[warpIdx].baseId);
				break;
                                default: {
                                	printf("kernel Type not found\n");
                                        return;
                                }
                        }

#if 1			
			if((threadIdx.x & 0x1f) == 0){
#if 1
				if((atomicSub((int*)&taskArgs[warpPool[warpIdx].taskId].doneGPU,1)) ==1){
					taskArgs[warpPool[warpIdx].taskId].doneHost = 0;
					atomicAdd((int*)&totalExecTasks[0],1);
//					printf("Kernel:%d\n", *totalExecTasks);
				}
#endif
				warpPool[warpIdx].warpId = 0;
				warpQ->contents[warpPool[warpIdx].queueId][warpPool[warpIdx].locId] = 0;
//				__threadfence();
				__threadfence_block();
			}
#endif
		}
#endif
	}
//	}
}



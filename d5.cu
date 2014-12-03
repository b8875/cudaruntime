#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BLOCKS 12
#define BLOCKSIZE 1024
//#define BSize 32
//#define QSize (BLOCKS*BLOCKSIZE)/BSize/32
#define BSize 24
#define QSize 16

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
int funcId;
};

typedef struct {
int contents[BSize][QSize]; // body of queue
int last[BSize]; // position of last element
}queue;

__device__ void init_queue(queue *q){
        int tid = blockDim.x*blockIdx.x+threadIdx.x;

        if(tid < BSize){
                // start form 1, since 1st warp is used in scheduling
                if(tid == 0){
                        q->last[tid] = 1;
                }else{
                        q->last[tid] = 0;
                }

                for (int i = 0; i < QSize; i++){
                        q->contents[tid][i] = 0;
                }

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
	int warp;
	int j;
	if(tid < 32){
		if(tid < BSize)
                        init_queue(warpQ);
		warp = 0;
		j = 0;
		while(!(*done)){
			if(*done) continue;
			if(tid < BSize){
				if(taskBuffer[tid].req == 1 && !(*done)){
					warp = taskBuffer[tid].warp;
					while(warp > 0){
						if(warpQ->contents[tid][warpQ->last[tid]] == 0){
//							printf("Scheduling:%d, %d\n", taskBuffer[tid].taskId, tid);
							warpPool[tid*QSize + warpQ->last[tid]].queueId = tid;
                                                        warpPool[tid*QSize + warpQ->last[tid]].locId = warpQ->last[tid];
							warpQ->contents[tid][warpQ->last[tid]] = 1;
							warpPool[tid*QSize + warpQ->last[tid]].baseId = j*32;
							warpPool[tid*QSize + warpQ->last[tid]].taskId = taskBuffer[tid].taskId;
							warpPool[tid*QSize + warpQ->last[tid]].warpId = 1;
							__threadfence();
							warp--;
							j++;
						}
						warpQ->last[tid]++;
                                                if(warpQ->last[tid] == QSize){
                                                        if(tid == 0){
                                                                warpQ->last[tid] = 1;
                                                        }else{
                                                                warpQ->last[tid] = 0;
                                                        }
                                                }

					
					}
				
					taskBuffer[tid].req = 0;
					j = 0;
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
			//		if((threadIdx.x & 0x1f) == 0) printf("Before:%d, %d\n", warpPool[warpIdx].taskId, warpPool[warpIdx].baseId);
				MatMul_kernel((int*)taskArgs[warpPool[warpIdx].taskId].A, (int*)taskArgs[warpPool[warpIdx].taskId].B, (int*)taskArgs[warpPool[warpIdx].taskId].C, taskArgs[warpPool[warpIdx].taskId].size, taskArgs[warpPool[warpIdx].taskId].size, taskArgs[warpPool[warpIdx].taskId].size, warpPool[warpIdx].baseId);
//                                      if((threadIdx.x & 0x1f) == 0) printf("After:%p\n", taskArgs[warpPool[warpIdx].taskId].C);
				break;
                                default: {
                                	printf("kernel Type not found\n");
                                        return;
                                }
                        }

#if 1			
			if((threadIdx.x & 0x1f) == 0){
				if((atomicSub((int*)&taskArgs[warpPool[warpIdx].taskId].doneGPU,1)) ==1){
					taskArgs[warpPool[warpIdx].taskId].doneHost = 0;
					atomicAdd((int*)&totalExecTasks[0],1);
//					printf("Kernel:%d\n", *totalExecTasks);
				}
				warpPool[warpIdx].warpId = 0;
				warpQ->contents[warpPool[warpIdx].queueId][warpPool[warpIdx].locId] = 0;
				__threadfence();
			}
#endif
		}
#endif
	}
//	}
}



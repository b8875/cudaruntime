#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCKS 12
#define BLOCKSIZE 1024
#define BSize 32
#define QSize (BLOCKS*BLOCKSIZE)/BSize/32

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
int *A, *B, *C;
int size;
int warpId;
int baseId;
int queueId;
int locId;
int taskId;
int funcId;
};

struct task_arg{
volatile int doneHost;
int doneGPU;
};


typedef struct {
int contents[BSize][QSize]; // body of queue
int first[BSize]; // position of first element
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
	}
        if (tid < 32) { 
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

//			C[0] = sum;
                }
        }
#if 0
	if((threadIdx.x & 0x1f) == 0){
		printf("Mat:%d\n", done);
	}
#endif
#endif
}

__global__ void deviceRT(volatile int *done, queue *warpQ,  volatile struct kernel_para *para, struct kernel_para *taskArgs, struct kernel_para_GPU *warpPool, int totalwarps){
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int i;
	__shared__ int j;
	volatile __shared__ int warp;
	int threadDone;
	int taskCount = 0;
	if (tid < 32){
		init_queue(warpQ);
		i = 0;
		j = 0;
		warp = 0;
		while(!(*done)){
			while((para[i].req == 0) &&!(*done)){
				//i++;
				//if(i == BSize) i = 0;
                        } 
			if(*done) {
				continue;
			}
			warp = para[i].warp;
//			if(tid == 0) printf("Before Scheduling:%d, %d\n", i, para[i].warp);

#if 1
#if 1
			if(warp > 0) {
//				if (tid == 0) printf("Kernel:%d, %d\n", i, para[i].taskId);
				taskCount++;
			}else if(!(*done)){
			printf("Something wrong, req:%d,paraWarp:%d, warp:%d,i:%d, ID:%d, tid:%d\n", para[i].req, para[i].warp, warp,i, para[i].taskId, tid);
			}
#endif
			threadDone = 0;
			while(warp > 0){
				int warpFound=0;
				int warpSched=0;
				if(warpQ->contents[tid][warpQ->last[tid]] == 0){
                                	warpFound = 1;
					if(atomicSub((int*)&warp,1) > 0){
						warpSched=1;
//						printf("Schedule:%d\n", para[i].taskId);
#if 1	
						warpPool[tid*QSize + warpQ->last[tid]].queueId = tid;
                                                warpPool[tid*QSize + warpQ->last[tid]].locId = warpQ->last[tid];
						warpPool[tid*QSize + warpQ->last[tid]].baseId = atomicAdd(&j,1)*BSize;

                                                warpQ->contents[tid][warpQ->last[tid]] = 1;
	
						warpPool[tid*QSize + warpQ->last[tid]].taskId = para[i].taskId;
                                                warpPool[tid*QSize + warpQ->last[tid]].warpId = 1;
						//taskArgs[para[i].taskId].doneHost = 0;
						__threadfence();
#endif
					}	
				}
				if( (warpFound==1 && warpSched==1) || (warpFound == 0) ) {
					warpQ->last[tid]++;
                                        if(warpQ->last[tid] == QSize){
                                                if(tid == 0){
                                                       	warpQ->last[tid] = 1;
                                                }else{
                                                        warpQ->last[tid] = 0;
                                                }
					}

                                }

			}
			threadDone=1;
			while(__all(threadDone == 1) == 0 ) ;
			//if(tid == 0) printf("Kernel After Scheduling:%d, %d, %d\n", i, para[i].warp, taskCount);
				para[i].req = 0;
				j = 0;
				i++;
				if(i == BSize) i = 0;
#endif
		}
#if 1	
			if(tid == 0) {
				printf("Scheduled tasks on the device %d\n", taskCount);
			}
#endif
	}else{

		int warpIdx = (blockDim.x*blockIdx.x+threadIdx.x)/BSize;
#if 1
                while(!(*done)){
                        while(warpPool[warpIdx].warpId == 0 && !(*done)){
                                //printf("BlockId", %warpIdx);
                        }
                        if(*done) {
                                return;
                        }
			
			switch(taskArgs[warpPool[warpIdx].taskId].funcId){
                                case 1:
                              MatMul_kernel((int*)taskArgs[warpPool[warpIdx].taskId].A, (int*)taskArgs[warpPool[warpIdx].taskId].B, (int*)taskArgs[warpPool[warpIdx].taskId].C, taskArgs[warpPool[warpIdx].taskId].size, taskArgs[warpPool[warpIdx].taskId].size, taskArgs[warpPool[warpIdx].taskId].size, warpPool[warpIdx].baseId);

                                        break;
                                default: {
                                        printf("kernel Type not found\n");
                                        return;
                                }
                        }

			if((threadIdx.x & 0x1f) == 0){
				atomicSub((int*)&taskArgs[warpPool[warpIdx].taskId].doneGPU,1);
				if(taskArgs[warpPool[warpIdx].taskId].doneGPU == 0){
					taskArgs[warpPool[warpIdx].taskId].doneHost = 0;
				}
				warpPool[warpIdx].warpId = 0;
				warpQ->contents[warpPool[warpIdx].queueId][warpPool[warpIdx].locId] = 0;

			}

		}
#if 0
                if((threadIdx.x & 0x1f) == 0 ) {
			printf("Total warps executed %d\n", *totalExecWarps);
		}
#endif
#endif
	}
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "para.h"

struct kernel_para{
volatile int *A, *B;
volatile int *C;
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
int taskId;
};

__device__ void init_queue(struct kernel_para_GPU *warpPool){
	int warpIdxx = (blockIdx.x*blockDim.x+threadIdx.x)/32;
	if((threadIdx.x) != 0){
		warpPool[warpIdxx+threadIdx.x].warpId = 0;
	}else{
		warpPool[warpIdxx+threadIdx.x].warpId = 1;
	}
	
		
}

__device__ void MatMul_kernel(int *A, int *B, int *C, int Size, int baseTid){
#if 1
        int row = baseTid + (threadIdx.x & 0x1f);
        for (int j = 0; j < Size; j++){
        	int sum = 0;
                for (int k = 0; k < Size; k++){
                	int a = A[row * Size + k];
                        int b = B[k * Size + j];
                        	sum += a * b;

                }
		C[row * Size + j] = sum;
        }
#endif
}

__device__ void VecAdd_kernel(int *A, int *B, int *C, int size, int baseTid)
{
    int i = baseTid + (threadIdx.x & 0x1f);
                //printf("In vec add with tid %d from block %d\n",i, blockIdx.x);
//                for(int j=0; j<200000; j++)
    if (i < size)
        C[i] = A[i] + B[i];
}


__global__ void deviceRT(volatile int *done, volatile int *totalExecTasks, struct kernel_para_GPU *warpPool, volatile struct kernel_para *taskBuffer, struct kernel_para *taskArgs, volatile int *exec, volatile int *totalScheTasks){
	int warpIdxx = (blockIdx.x*blockDim.x + threadIdx.x)/32;
	__shared__ int warp;
	__shared__ int warpQPointer;
	__shared__ int base;
	int threadDone;
	// Init warp queue contents and pointers
#if 1
	if(threadIdx.x < QSize){
		init_queue(warpPool);
		warp = 0;
		warpQPointer = 0;
		base = 0;
	}
	__syncthreads();
#endif
	// scheduling in master warps
	if(threadIdx.x < 32) {
		while(!(*done)){
			if(threadIdx.x < QSize){
	//		if(*done) return;
				if(taskBuffer[warpQPointer*BSize+blockIdx.x].req == 1 && !(*done)){
					warp = taskBuffer[warpQPointer*BSize+blockIdx.x].warp;
					// search free warps
					while(1){
						threadDone = 0;
						if(warpPool[warpIdxx+threadIdx.x].warpId == 0){
							if(atomicSub(&warp, 1) > 0){
								warpPool[warpIdxx+threadIdx.x].taskId = taskBuffer[warpQPointer*BSize+blockIdx.x].taskId;
								warpPool[warpIdxx+threadIdx.x].baseId = atomicAdd(&base, 1)*32;
								warpPool[warpIdxx+threadIdx.x].warpId = 1;
								__threadfence_block();
							}// End if(warp > 0)
						}// End if (warpQ->contents)
						if(warp <= 0){
							threadDone = 1;
						}
						if(__all(threadDone == 1) != 0){
							if(threadIdx.x == 0){
								taskBuffer[warpQPointer*BSize+blockIdx.x].req = 0;
								base = 0;
		//						atomicAdd((int*)&totalExecTasks[0],1);
			                                        }
			                                	break;

						}// End warp vote
					}//End while(1)
				} // End taskBuffer if
				if(threadIdx.x == 0) {
					warpQPointer++;
                	if(warpQPointer == SSize){
						warpQPointer = 0;
					}
				}
			}// End while(!(*done))
		}//End if(threadIdx.x < 32)
	}
#if 1
	else{
#if 1
		while(!(*exec)){
//			if(*exec) return;
			if(warpPool[warpIdxx].warpId == 1 && !(*exec)){

				MatMul_kernel((int*)taskArgs[warpPool[warpIdxx].taskId].A, (int*)taskArgs[warpPool[warpIdxx].taskId].B, (int*)taskArgs[warpPool[warpIdxx].taskId].C, taskArgs[warpPool[warpIdxx].taskId].size, warpPool[warpIdxx].baseId);

				if((threadIdx.x & 0x1f) == 0){
					if((atomicSub((int*)&taskArgs[warpPool[warpIdxx].taskId].doneGPU,1)) ==1){
                                      		taskArgs[warpPool[warpIdxx].taskId].doneHost = 0;
                                        	//atomicAdd((int*)&totalExecTasks[blockIdx.x],1);
						atomicAdd((int*)&totalScheTasks[0],1);
                                	}

					warpPool[warpIdxx].warpId = 0;
					__threadfence_block();
				}
			}
		}
#endif
	}// End else
#endif
}



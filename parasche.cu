#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>

#include "para.h"

#define imin(a, b) (a<=b?a:b)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

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

int ipow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}

extern __global__ void deviceRT(volatile int *done, volatile int *totalExecTasks, struct kernel_para_GPU *warpPool, volatile struct kernel_para *taskBuffer, struct kernel_para *taskArgs, volatile int *exec, volatile int *totalScheTasks);
int main(int argc, char** argv){
        double startTime, endTime;
        int totalWarps = ((BLOCKSIZE*BLOCKS)/32);
	cudaSetDevice(0);
        cudaDeviceReset();

	cudaStream_t s2;
	cudaStream_t s3;
	checkCudaErrors(cudaStreamCreate(&s2));
	checkCudaErrors(cudaStreamCreate(&s3));

	// To interrupt the runtime
        int *done, *doneDev;
	int *exec, *execDev;
	int *totalExecTasks, *totalExecTasksDev;
	int *totalScheTasks, *totalScheTasksDev;
	struct kernel_para_GPU *warpPoolDev;
	struct kernel_para *taskArgs, *taskArgsDev;
	struct kernel_para *taskparaBuffer, *taskparaBufferDev;

	checkCudaErrors(cudaHostAlloc(&totalScheTasks, sizeof(int), cudaHostAllocDefault));
        checkCudaErrors(cudaMalloc(&totalScheTasksDev, sizeof(int)));
	checkCudaErrors(cudaHostAlloc(&exec, sizeof(int), cudaHostAllocDefault));
        checkCudaErrors(cudaMalloc(&execDev, sizeof(int)));


	// done flag
        checkCudaErrors(cudaHostAlloc(&done, sizeof(int), cudaHostAllocDefault));
        checkCudaErrors(cudaMalloc(&doneDev, sizeof(int)));

        checkCudaErrors(cudaMalloc(&warpPoolDev, totalWarps*sizeof(struct kernel_para_GPU)));

	checkCudaErrors(cudaHostAlloc(&totalExecTasks, BLOCKS*sizeof(int), cudaHostAllocDefault));
        checkCudaErrors(cudaMalloc(&totalExecTasksDev, BLOCKS*sizeof(int)));

	checkCudaErrors(cudaHostAlloc(&taskArgs, tasks*sizeof(struct kernel_para), cudaHostAllocDefault));
        checkCudaErrors(cudaMalloc(&taskArgsDev, tasks*sizeof(struct kernel_para)));
	
	checkCudaErrors(cudaHostAlloc(&taskparaBuffer, BSize*SSize*sizeof(struct kernel_para), cudaHostAllocDefault));
        checkCudaErrors(cudaMalloc(&taskparaBufferDev, BSize*SSize*sizeof(struct kernel_para)));

	// input data
        int *aDev[tasks], *bDev[tasks], *cDev[tasks];
        int *a[tasks], *b[tasks], *c[tasks];

        for(int i=0; i<tasks; i++) {
                checkCudaErrors(cudaMalloc(&aDev[i], N*sizeof(int)));
                checkCudaErrors(cudaMalloc(&bDev[i], N*sizeof(int)));
                checkCudaErrors(cudaMalloc(&cDev[i], N*sizeof(int)));
                checkCudaErrors(cudaHostAlloc(&a[i], N*sizeof(int), NULL));
                checkCudaErrors(cudaHostAlloc(&b[i], N*sizeof(int), NULL));
                checkCudaErrors(cudaHostAlloc(&c[i], N*sizeof(int), NULL));
        }

        for(int i = 0; i < tasks; i++){
                for(int j=0; j<N; j++) {
                        a[i][j]= (i%32)+1;
                        b[i][j]= (i%32)+1;
                        c[i][j] = 0;
                }
        }


	for(int i = 0; i < tasks; i++){
                checkCudaErrors(cudaMemcpyAsync(aDev[i], a[i] , N*sizeof(int),cudaMemcpyHostToDevice, s3));
                checkCudaErrors(cudaMemcpyAsync(bDev[i], b[i] , N*sizeof(int),cudaMemcpyHostToDevice, s3));
                checkCudaErrors(cudaMemcpyAsync(cDev[i], c[i] , N*sizeof(int),cudaMemcpyHostToDevice, s3));
        }

        for(int i = 0; i < tasks; i++){
                 // init. task para
                taskArgs[i].A = aDev[i];
                taskArgs[i].B = bDev[i];
                taskArgs[i].C = cDev[i];
                taskArgs[i].size = DATASIZE;
                taskArgs[i].block = 1;
                taskArgs[i].thread = THREADS;
                taskArgs[i].warp = THREADS/32;
                taskArgs[i].funcId = 1;
        //        taskArgs[i].taskId = i;
                taskArgs[i].req = 1;
                taskArgs[i].doneHost = 1;
                taskArgs[i].doneGPU = THREADS/32;
//		printf("Host:%p\n", taskArgs[i].A);

        }

	for(int i = 0; i < (BSize*SSize); i++){
		taskparaBuffer[i].req = 0;
	}

	*done = 0;
	*exec = 0;
        *totalScheTasks = 0;
	for(int k=0; k< BLOCKS; k++)
        totalExecTasks[k] = 0;

	checkCudaErrors(cudaMemcpyAsync(execDev, exec, sizeof(int), cudaMemcpyHostToDevice, s3));
	checkCudaErrors(cudaMemcpyAsync(doneDev, done, sizeof(int), cudaMemcpyHostToDevice, s3));
	checkCudaErrors(cudaMemcpyAsync(totalExecTasksDev, totalExecTasks, BLOCKS*sizeof(int), cudaMemcpyHostToDevice, s3));
	checkCudaErrors(cudaMemcpyAsync(totalScheTasksDev, totalScheTasks, sizeof(int), cudaMemcpyHostToDevice, s3));
	checkCudaErrors(cudaMemcpyAsync(taskparaBufferDev, taskparaBuffer, BSize*SSize*sizeof(struct kernel_para), cudaMemcpyHostToDevice, s3));
	checkCudaErrors(cudaMemcpyAsync(taskArgsDev, taskArgs, tasks*sizeof(struct kernel_para), cudaMemcpyHostToDevice, s3));
	checkCudaErrors(cudaStreamSynchronize(s3));
	deviceRT<<<BLOCKS,BLOCKSIZE,0, s2>>>(doneDev, totalExecTasksDev, warpPoolDev, taskparaBufferDev, taskArgsDev, execDev, totalScheTasksDev);
#if 1
	// para delivery
	int j = 0;
	int c1 = 0;
	startTime = my_timer();
	while(j < tasks){
		for(int i = 0; i < (SSize*BSize); i++){
			if(taskparaBuffer[i].req == 0){
				taskparaBuffer[i].warp = THREADS/32;
				taskparaBuffer[i].req = 1;
				taskparaBuffer[i].taskId = j;
				//	printf("Host:%d, %d\n", i*BSize+j, t);
				checkCudaErrors(cudaMemcpyAsync(&taskparaBufferDev[i], &taskparaBuffer[i], sizeof(struct kernel_para), cudaMemcpyHostToDevice, s3));
				j++;
				if(j == tasks) break;
			}
		}
		if(j == tasks) break;
		checkCudaErrors(cudaMemcpyAsync(taskparaBuffer, taskparaBufferDev, BSize*SSize*sizeof(struct kernel_para), cudaMemcpyDeviceToHost, s3));
		checkCudaErrors(cudaStreamSynchronize(s3));
		c1++;
	}
	endTime = my_timer();
        printf("Elapsed Time1:%lf sec.\n", (endTime-startTime));
	printf("Iteration1:%d\n", c1);
#endif

#if 1
	int all = 0;
	startTime = my_timer();
	//int sum = 0;
	while(*totalScheTasks < tasks){
//		checkCudaErrors(cudaMemcpyAsync(totalExecTasks, totalExecTasksDev, BLOCKS*sizeof(int), cudaMemcpyDeviceToHost, s3));
		checkCudaErrors(cudaMemcpyAsync(totalScheTasks, totalScheTasksDev, sizeof(int), cudaMemcpyDeviceToHost, s3));
		checkCudaErrors(cudaStreamSynchronize(s3));
		//printf("Tasks done %d\n", *totalExecTasks);
#if 0
		sum = 0;
		for(int k=0; k< BLOCKS; k++) {
			sum += totalExecTasks[k];
		}
		
		if(all % 100 == 0)
		for(int k=0; k< BLOCKS; k++) {
			printf("Block %d tasks %d\n", k, totalExecTasks[k]);
		}
		
		//if( (double)sum > (double)(0.98*tasks)) 
			//break;
#endif
		all++;
		//if(all > 40000) break;
	}
	endTime = my_timer();
    	printf("Elapsed Time2:%lf sec.\n", (endTime-startTime));
	printf("Iterations:%d, %d\n", all, *totalScheTasks);
	

#endif
#if 1
	*exec = 1;
        checkCudaErrors(cudaMemcpyAsync(execDev, exec, sizeof(int), cudaMemcpyHostToDevice, s3));
	*done = 1;
	checkCudaErrors(cudaMemcpyAsync(doneDev, done, sizeof(int), cudaMemcpyHostToDevice, s3));

#endif
#if 1
	  // copy back results of tasks
        for(int i=0; i<tasks; i++) {
                checkCudaErrors(cudaMemcpyAsync (c[i], cDev[i] , N*sizeof(int),cudaMemcpyDeviceToHost, s3));
        }
        checkCudaErrors(cudaStreamSynchronize(s3));
#endif

#if 1
        // verification
        for (int i = 0; i < tasks; i++){
                for(int j = 0; j < N; j++){
                        if(c[i][j] != DATASIZE*ipow((i%32)+1, 2)){
                                printf("Error:%d, %d\n", i, c[i][j]);
                                break;
                        }
                }
        }
#endif

	 for(int i = 0; i < tasks; i++){
                checkCudaErrors(cudaFreeHost(a[i]));
                checkCudaErrors(cudaFreeHost(b[i]));
                checkCudaErrors(cudaFreeHost(c[i]));
                checkCudaErrors(cudaFree(aDev[i]));
                checkCudaErrors(cudaFree(bDev[i]));
                checkCudaErrors(cudaFree(cDev[i]));
        }

	checkCudaErrors(cudaStreamDestroy(s2));
	cudaStreamDestroy(s3);

	cudaFreeHost(done);
	cudaFreeHost(exec);
	cudaFreeHost(totalExecTasks);
	cudaFreeHost(taskArgs);
	cudaFreeHost(taskparaBuffer);
	cudaFreeHost(totalScheTasks);

	cudaFree(totalExecTasksDev);
	cudaFree(totalScheTasksDev);
	cudaFree(warpPoolDev);
	cudaFree(doneDev);
	cudaFree(execDev);
	cudaFree(taskArgsDev);
	cudaFree(taskparaBufferDev);
	return 0;
}



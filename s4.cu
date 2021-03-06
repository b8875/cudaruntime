#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>

#define BLOCKS 12
#define BLOCKSIZE 1024
//#define BSize 32
//#define QSize (BLOCKS*BLOCKSIZE)/BSize/32
#define BSize 12
#define QSize 32
#define DATASIZE 32
#define THREADS 32
#define N (DATASIZE*DATASIZE)
#define tasks 24

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

extern __global__ void deviceRT(volatile int *done, queue *warpQ, volatile struct kernel_para *para, volatile struct kernel_para *taskArgs, volatile struct kernel_para_GPU *warpPool, int totalwarps, volatile int *totalExecWarps);
//extern __global__ void deviceRT(volatile int *done, volatile int *totalExecTasks, volatile kernel_para_GPU *warpPool, volatile struct kernel_para *taskBuffer, struct kernel_para *taskArgs, queue *warpQ);
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

int main(int argc, char** argv){
        double startTime, endTime;
        int totalwarps = ((BLOCKSIZE*BLOCKS)/32);

	cudaStream_t s1;
        cudaStream_t s2;
	cudaStream_t s3[BSize];
	cudaStream_t s4;
	checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaDeviceReset());


	checkCudaErrors(cudaStreamCreate(&s1));
        checkCudaErrors(cudaStreamCreate(&s2));
	checkCudaErrors(cudaStreamCreate(&s4));
	for(int i = 0; i < BSize; i++){
		checkCudaErrors(cudaStreamCreate(&s3[i]));
	}

	cudaEvent_t event1;
	checkCudaErrors(cudaEventCreate(&event1));
	// To interrupt the runtime
	int *done, *doneDev;
	// para buffer
	struct kernel_para *paraBuffer, *paraBufferDev;
	// warp pool in device to track free warps
        struct kernel_para_GPU *warpPool, *warpPoolDev;
	// para of task
	struct kernel_para *taskArgs, *taskArgsDev;
	// warp queue
	queue *warpQ;

#if 1	
	// done flag
        checkCudaErrors(cudaHostAlloc(&done, sizeof(int), cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&doneDev, sizeof(int)));
#if 1	
	int *totalExecWarps, *totalExecWarpsDev;
	checkCudaErrors(cudaHostAlloc(&totalExecWarps, sizeof(int), cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&totalExecWarpsDev, sizeof(int)));
#endif
	// para buffer
	checkCudaErrors(cudaMalloc(&paraBufferDev, BSize*sizeof(struct kernel_para)));
        checkCudaErrors(cudaHostAlloc(&paraBuffer, BSize*sizeof(struct kernel_para), NULL));
	// warp Pool in device
	checkCudaErrors(cudaMalloc(&warpPoolDev, totalwarps*sizeof(struct kernel_para_GPU)));
	checkCudaErrors(cudaHostAlloc(&warpPool, totalwarps*sizeof(struct kernel_para_GPU), NULL));
	// warp queue
	checkCudaErrors(cudaMalloc(&warpQ, sizeof(queue)));
	// para of tasks
	checkCudaErrors(cudaMalloc(&taskArgsDev, tasks*sizeof(struct kernel_para)));
        checkCudaErrors(cudaHostAlloc(&taskArgs, tasks*sizeof(struct kernel_para), NULL));
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

	for(int i = 0; i < totalwarps; i++){
		warpPool[i].warpId = 0;
	}
	// Init. of para buffer
	for(int i = 0; i < BSize; i++){
                paraBuffer[i].req = 0;
        }
	*done = 0;
	*totalExecWarps = 0;
	checkCudaErrors(cudaMemcpyAsync (totalExecWarpsDev, totalExecWarps, sizeof(int), cudaMemcpyHostToDevice, s1));
        checkCudaErrors(cudaMemcpyAsync (doneDev, done, sizeof(int), cudaMemcpyHostToDevice, s1));
        checkCudaErrors(cudaStreamSynchronize(s1));


	for(int i = 0; i < tasks; i++){
                checkCudaErrors(cudaMemcpyAsync(aDev[i], a[i] , N*sizeof(int),cudaMemcpyHostToDevice, s1));
                checkCudaErrors(cudaMemcpyAsync(bDev[i], b[i] , N*sizeof(int),cudaMemcpyHostToDevice, s1));
                checkCudaErrors(cudaMemcpyAsync(cDev[i], c[i] , N*sizeof(int),cudaMemcpyHostToDevice, s1));
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
		taskArgs[i].taskId = i;
		taskArgs[i].req = 1;
                taskArgs[i].doneHost = 1;
                taskArgs[i].doneGPU = THREADS/32; 

	}
	checkCudaErrors(cudaMemcpyAsync(paraBufferDev, paraBuffer, BSize*sizeof(struct kernel_para),cudaMemcpyHostToDevice, s1));
	checkCudaErrors(cudaMemcpyAsync(taskArgsDev, taskArgs, tasks*sizeof(struct kernel_para),cudaMemcpyHostToDevice, s1));
	checkCudaErrors(cudaMemcpyAsync(warpPoolDev, warpPool, totalwarps*sizeof(struct kernel_para_GPU),cudaMemcpyHostToDevice, s1));
	checkCudaErrors(cudaStreamSynchronize(s1));

	deviceRT<<<BLOCKS,BLOCKSIZE,0, s2>>>(doneDev, warpQ, paraBufferDev, taskArgsDev, warpPoolDev, totalwarps, totalExecWarpsDev);	     
//	deviceRT<<<BLOCKS,BLOCKSIZE,0, s2>>>(doneDev, totalExecWarpsDev, warpPoolDev, paraBufferDev, taskArgsDev, warpQ);

	printf("Enter task delivery\n");
	// critical section
	startTime = my_timer();
#endif	
	int j = 0;
	while(j < tasks){
#if 1
		for(int i = 0; i < BSize; i++){
			if(paraBuffer[i].req == 0){
//				printf("Host:%d, %d\n", i, j);
				paraBuffer[i].warp = THREADS/32;
				paraBuffer[i].taskId = taskArgs[j].taskId;
				paraBuffer[i].req = 1;
				checkCudaErrors(cudaMemcpyAsync(&paraBufferDev[i], &paraBuffer[i] , sizeof(struct kernel_para),cudaMemcpyHostToDevice, s1));
//				checkCudaErrors(cudaStreamSynchronize(s1));
				j++;
				if (j == tasks) break;
			}

		}
	//	printf("Done scheduling %d tasks\n", j);
		if(j == tasks) break;	
		checkCudaErrors(cudaMemcpyAsync(paraBuffer, paraBufferDev , BSize*sizeof(struct kernel_para),cudaMemcpyDeviceToHost, s1));
		checkCudaErrors(cudaStreamSynchronize(s1));
#endif
	
	}

	endTime = my_timer();
        printf("Elapsed Time1:%lf sec.\n", (endTime-startTime));
#if 1
	startTime = my_timer();
	int all = 0;
	while(*totalExecWarps < tasks){
//		while(all < 100000){
                checkCudaErrors(cudaMemcpyAsync(totalExecWarps, totalExecWarpsDev , sizeof(int),cudaMemcpyDeviceToHost, s1));
                checkCudaErrors(cudaStreamSynchronize(s1));
                all++;
        }

	endTime = my_timer();
        printf("Elapsed Time2:%lf sec.\n", (endTime-startTime));
	printf("Iteration:%d\n", all);
#endif
	*done = 1;
        checkCudaErrors(cudaMemcpyAsync(doneDev, done , sizeof(int),cudaMemcpyHostToDevice, s1));

#if 1
	// copy back results of tasks
        for(int i=0; i<tasks; i++) {
                cudaMemcpyAsync (c[i], cDev[i] , N*sizeof(int),cudaMemcpyDeviceToHost, s1);
        }
        cudaStreamSynchronize(s1);
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

//cleanup:
	// memory free
	for(int i = 0; i < tasks; i++){
		checkCudaErrors(cudaFreeHost(a[i]));
		checkCudaErrors(cudaFreeHost(b[i]));
		checkCudaErrors(cudaFreeHost(c[i]));
		checkCudaErrors(cudaFree(aDev[i]));
		checkCudaErrors(cudaFree(bDev[i]));
		checkCudaErrors(cudaFree(cDev[i]));
	}
	// stream free
	checkCudaErrors(cudaStreamDestroy(s1));
        checkCudaErrors(cudaStreamDestroy(s2));
	checkCudaErrors(cudaStreamDestroy(s4));
	for(int i = 0; i < BSize; i++){
		checkCudaErrors(cudaStreamDestroy(s3[i]));
	}
	// event free
	checkCudaErrors(cudaEventDestroy(event1));
	// host data free
	checkCudaErrors(cudaFreeHost(done));
	checkCudaErrors(cudaFreeHost(paraBuffer));
	checkCudaErrors(cudaFreeHost(taskArgs));
	checkCudaErrors(cudaFreeHost(warpPool));
//	checkCudaErrors(cudaFreeHost(totalExecWarps));

	// device data free
	checkCudaErrors(cudaFree(doneDev));
	checkCudaErrors(cudaFree(paraBufferDev));
	checkCudaErrors(cudaFree(taskArgsDev));
	checkCudaErrors(cudaFree(warpPoolDev));
        checkCudaErrors(cudaFree(warpQ));
//	checkCudaErrors(cudaFree(totalExecWarpsDev));

	return 0;
}

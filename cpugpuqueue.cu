/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 6
#define N_STREMS 64
#define HIST_SIZE 256

typedef unsigned char uchar;
#define OUT

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

#define QUEUE_SIZE 10


__device__ __host__ bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

__device__ __host__ uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

__device__  void gpu_image_to_histogram(uchar *image, int *histogram) {
    uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION, threadIdx.x % IMG_DIMENSION);
    atomicAdd(&histogram[pattern], 1);
}

__device__  void gpu_histogram_distance(int *h1, int *h2, double *distance) {
    int length = 256;
    int tid = threadIdx.x;
    if(tid<length){
    	distance[tid] = 0;
    	if (h1[tid] + h2[tid] != 0) {
    	    distance[tid] = ((double)SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
    	}
    	h1[tid] = h2[tid]=0;
    }
    __syncthreads();


    while (length > 1) {
        if (threadIdx.x < length / 2) {
            distance[tid] = distance[tid] + distance[tid + length / 2];
        }
        length /= 2;
        __syncthreads();
    }
}


void image_to_histogram(uchar *image, int *histogram) {
    memset(histogram, 0, sizeof(int) * 256);
    for (int i = 0; i < IMG_DIMENSION; i++) {
        for (int j = 0; j < IMG_DIMENSION; j++) {
            uchar pattern = local_binary_pattern(image, i, j);
            histogram[pattern]++;
        }
    }
}

double histogram_distance(int *h1, int *h2) {
    /* we'll use the chi-square distance */
    double distance = 0;
    for (int i = 0; i < 256; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}
/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}


/*************************************************/
/*******CLASS***producer***consumer****queue******/
/*************************************************/
class cpu2gpuQueue {
public:
	cpu2gpuQueue():size(QUEUE_SIZE),head(0),tail(0){/*printf("head=%d\tsize=%d\n",head,size)*/;}
	~cpu2gpuQueue(){}
	__device__ __host__ cpu2gpuQueue& operator=(const cpu2gpuQueue& rhs);
	__host__ int produce(uchar* imag1,uchar* imag2);
	__device__ int consume(uchar* images);



private:
	volatile int size;
	volatile int head;
	volatile int tail;
	uchar q[QUEUE_SIZE*SQR(IMG_DIMENSION)];
};

__device__ __host__ cpu2gpuQueue& cpu2gpuQueue::operator=(const cpu2gpuQueue& rhs)
{
	this->head=rhs.head;
	this->size=rhs.size;
	this->tail=rhs.tail;
	memcpy(this->q,rhs.q,QUEUE_SIZE*SQR(IMG_DIMENSION)*sizeof(*rhs.q));
	return *this;
}
__device__ int cpu2gpuQueue::consume(uchar* images)
{
	if(!threadIdx.x)
	{
		printf("cpu2gpuQueue::consume\n");
		printf("tail=%d\thead=%d\n",tail,head);
	}
	if(!(tail<head))return 0;
	/*int i;
	for(i=threadIdx.x;i<2*SQR(IMG_DIMENSION);i+=gridDim.x)
		images[i]=q[(tail%QUEUE_SIZE)*2*SQR(IMG_DIMENSION)+i];*/
	//make sure all threads copied before increasing the value of tail
	 __syncthreads();
	 if(!threadIdx.x)
	 {
		 size++;
		 tail++;
		 printf("gpu size=%d\n",size);
		 __threadfence_system();
	 }
	// __syncthreads();
	return 1;
}
__host__ int cpu2gpuQueue::produce(uchar* imag1,uchar* imag2)
{
	if(!(head<size)){
		//printf("head=%d\tsize=%d\ttrue\n",head,size);
		return 0;

	}
	memcpy(&q[(head%QUEUE_SIZE)*2*SQR(IMG_DIMENSION)],imag1,SQR(IMG_DIMENSION)*sizeof(uchar));
	memcpy(&q[(head%QUEUE_SIZE)*2*SQR(IMG_DIMENSION)+SQR(IMG_DIMENSION)],imag2,SQR(IMG_DIMENSION)*sizeof(uchar));
	head++;
	return 1;
}

class gpu2cpuQueue {
public:
	gpu2cpuQueue():size(QUEUE_SIZE),head(0),tail(0){}
	~gpu2cpuQueue(){}
	__device__ __host__ gpu2cpuQueue& operator=(const gpu2cpuQueue& rhs);
	__device__ int produce(double distance);
	__host__ int consume(double* distance);
	void testadd(){size++;}
	void test(){
		for(int i=0;i<QUEUE_SIZE;i++)
			printf("%f\t",q[i]);
		printf("\n");
	}
private:
	volatile int size;
	volatile int head;
	volatile int tail;
	double q[QUEUE_SIZE];
};
__device__ __host__ gpu2cpuQueue& gpu2cpuQueue::operator=(const gpu2cpuQueue& rhs)
{
	this->head=rhs.head;
	this->size=rhs.size;
	this->tail=rhs.tail;
	memcpy(this->q,rhs.q,QUEUE_SIZE*sizeof(*rhs.q));
	return *this;
}
static int x=0;
__host__ int gpu2cpuQueue::consume(double* distance)
{


	if((!(x%100))&&x<500){
		printf("gpu2cpuQueue::consume\n");
		printf("tail=%d\thead=%d\n",tail,head);
	}
	x++;


	if(!(tail<head))return 0;
	*distance=q[(tail%QUEUE_SIZE)];
	printf("gpu2cpuQueue::consume\n");
	printf("tail=%d\thead=%d\n",tail,head);
	printf("distance=%f\n",*distance);
	printf("size=%d\n",size);
	size++;
	tail++;
	return 1;
}
__device__ int gpu2cpuQueue::produce(double distance)
{

	if(!(head<size)) return 0;
	if(threadIdx.x) return 1;
	printf("gpu size=%d\t(head%QUEUE_SIZE)=%d\n",size,(head%QUEUE_SIZE));
	q[(head%QUEUE_SIZE)]=distance;
	//printf("before\n");
	//printf("distance=%f\n",distance);

	__threadfence_system();
	//printf("after\n");
	head++;
	__threadfence_system();
	return 1;
}
struct QP{
	cpu2gpuQueue cpugpu;
	gpu2cpuQueue gpucpu;
};
__global__ void test(struct QP* Ptr){
	int i;
	if(!threadIdx.x) printf("test kernel\n");
	__shared__ uchar images[2*SQR(IMG_DIMENSION)];
	__shared__ int hist1[HIST_SIZE],hist2[HIST_SIZE];
	__shared__ double distance[HIST_SIZE];

	//if(threadIdx.x<HIST_SIZE)
		//hist1[threadIdx.x]=hist2[threadIdx.x]=0;

	//if(!threadIdx.x) printf("test kernel\n");
	for(int i=0;i<NREQUESTS;i++)
	{

		//if(!threadIdx.x)printf("gpu loop i=%d\n",i);
		while(!Ptr->cpugpu.consume(images));
		if(!threadIdx.x)printf("gpu loop i=%d\n",i);
		//gpu_image_to_histogram(images,hist1);
		//gpu_image_to_histogram(images+SQR(IMG_DIMENSION),hist2);
		//__syncthreads();
		//gpu_histogram_distance(hist1,hist2,distance);
		//while(!Ptr->gpucpu.produce(distance[0]));
		while(!Ptr->gpucpu.produce((double)i));
		__syncthreads();
	}
}



__global__ void test1(int* Ptr)
{
	int i;
	for(int i=0;i<NREQUESTS;++i)
		Ptr[i]=i;
	printf("kernel finish\n");
	return;
}

int main(void) {
	uchar *images1;
	uchar *images2;
	CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
	CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
	load_image_pairs(images1, images2);
	double t_start, t_finish;
	double total_distance=0,distance=0;
	int i=NREQUESTS,finished=0;
	struct QP *cpuqp,*gpuqp;
	CUDA_CHECK( cudaHostAlloc(&cpuqp, sizeof(struct QP), cudaHostAllocWriteCombined) );
	cpuqp->cpugpu=cpu2gpuQueue();
	cpuqp->gpucpu=gpu2cpuQueue();
	CUDA_CHECK( cudaHostGetDevicePointer(&gpuqp,cpuqp,0) );


    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    t_start  = get_time_msec();
    for (int i = 0; i < NREQUESTS; i++) {
        int img_idx = i % N_IMG_PAIRS;
        image_to_histogram(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    total_distance=0;
	test<<<1, 1024>>>(gpuqp);
	cpuqp->gpucpu.testadd();


	//printf("after\n");
	for(int i=0;i<NREQUESTS;i++)
	{
		printf("cpu loop i=%d\n",i);
		distance=0;
		if(cpuqp->gpucpu.consume(&distance))
		{
			//printf("distance=%f\n",distance);
			total_distance+=distance;
			finished++;
			printf("finished=%d\n",finished);

		}
		int img_idx = i % N_IMG_PAIRS,j;
		//for(j=0;j<SQR(IMG_DIMENSION);++j)printf("%d%d",images1[img_idx * IMG_DIMENSION * IMG_DIMENSION+j],images2[img_idx * IMG_DIMENSION * IMG_DIMENSION+j]);
		//printf("\n");
		while(!cpuqp->cpugpu.produce(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION],&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION]));
		cpuqp->gpucpu.test();
	}
	printf("finish loop\n");

	/*while(finished<NREQUESTS)
	{
		if(cpuqp->gpucpu.consume(&distance))
		{
			//printf("distance=%f\n",distance);
			total_distance+=distance;
			finished++;
			printf("finished=%d\n",finished);

		}
	}*/

	printf("finish loop 2\n");
	CUDA_CHECK( cudaDeviceSynchronize());
	cpuqp->gpucpu.test();
	printf("average distance between images %f\n", total_distance / NREQUESTS);
	return 0;
}

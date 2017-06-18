#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 1000
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





class cpu2gpuQueue {
public:
	cpu2gpuQueue():size(QUEUE_SIZE),head(0),tail(0){/*printf("head=%d\tsize=%d\n",head,size)*/;}
	~cpu2gpuQueue(){}
	//__device__ __host__ cpu2gpuQueue& operator=(const cpu2gpuQueue& rhs);
	__host__ int produce(int img_idx);
	__device__ int consume(int* img_idx);



private:
	volatile int size;
	volatile int head;
	volatile int tail;
	int q[QUEUE_SIZE];
};




__device__ int cpu2gpuQueue::consume(int* img_idx)
{
	if(!(tail<head))return 0;
	*img_idx=q[(tail%QUEUE_SIZE)];
	size++;
	tail++;
	__threadfence_system();
	return 1;
}


__host__ int cpu2gpuQueue::produce(int img_idx)
{
	if(!(head<size)){
		return 0;

	}
	q[(head%QUEUE_SIZE)]=img_idx;
	head++;
	return 1;
}


class gpu2cpuQueue {
public:
	gpu2cpuQueue():size(QUEUE_SIZE),head(0),tail(0){}
	~gpu2cpuQueue(){}
	//__device__ __host__ gpu2cpuQueue& operator=(const gpu2cpuQueue& rhs);
	__device__ int produce(double distance);
	__host__ int consume(double* distance);

private:
	volatile int size;
	volatile int head;
	volatile int tail;
	double q[QUEUE_SIZE];
};
/*__device__ __host__ gpu2cpuQueue& gpu2cpuQueue::operator=(const gpu2cpuQueue& rhs)
{
	this->head=rhs.head;
	this->size=rhs.size;
	this->tail=rhs.tail;
	memcpy(this->q,rhs.q,QUEUE_SIZE*sizeof(*rhs.q));
	return *this;
}*/
__host__ int gpu2cpuQueue::consume(double* distance)
{
	if(!(tail<head))return 0;
	*distance=q[(tail%QUEUE_SIZE)];
	size++;
	tail++;
	return 1;
}
__device__ int gpu2cpuQueue::produce(double distance)
{
	if(!(head<size)) return 0;
	q[(head%QUEUE_SIZE)]=distance;
	__threadfence_system();
	head++;
	__threadfence_system();
	return 1;
}

typedef struct {
	cpu2gpuQueue cpugpu;
	gpu2cpuQueue gpucpu;
} QP;



__global__ void test(QP** gpuQPs,uchar* imags1,uchar* imags2 ){
	if(!threadIdx.x) printf("test kernel\n");
	//__shared__ uchar images[2*SQR(IMG_DIMENSION)];
	__shared__ int hist1[HIST_SIZE],hist2[HIST_SIZE];
	__shared__ double distance[HIST_SIZE];
	__shared__ double total_distance;
	__shared__ int img_idx;

	if(!threadIdx.x) total_distance=0;


	for(int i=threadIdx.x;i<HIST_SIZE;i+=gridDim.x)
		hist1[i]=hist2[i]=0;

	bool running=true;
	//for(int i=0;i<NREQUESTS;i++)
	while( running )
	{
		if(!threadIdx.x)while(!gpuQPs[blockIdx.x]->cpugpu.consume(&img_idx));
		__syncthreads();
		//printf("img_idx=%d\n",img_idx);
		if(img_idx==-1)
			break;
		gpu_image_to_histogram(&imags1[img_idx*SQR(IMG_DIMENSION)],hist1);
		gpu_image_to_histogram(&imags2[img_idx*SQR(IMG_DIMENSION)],hist2);
		__syncthreads();
		gpu_histogram_distance(hist1,hist2,distance);
		__syncthreads();
		if(!threadIdx.x)total_distance+=distance[0];
		if(!threadIdx.x)while(!gpuQPs[blockIdx.x]->gpucpu.produce(distance[0]));
		__syncthreads();
	}
	__syncthreads();
	if(!threadIdx.x)printf("gpu average distance between images %f\n", total_distance / NREQUESTS);
}


int calcNumOfThreadblocks(){return 2;}
void checkQueueComplition(int num_of_threadblocks,QP **cpuQPs,int * finished, double* total_distance )
{

	double distance;
	for(int i=0,ret;i<num_of_threadblocks;i++)
	{
		do{
			ret=cpuQPs[i]->gpucpu.consume(&distance);
			*total_distance+=ret*distance;
			*finished+=ret;
		}while(ret);
	}
}
void QueueProduce(int num_of_threadblocks,QP **cpuQPs,int img_idx,int * finished, double* total_distance )
{
	bool produced=false;
	while(!produced)
	{
		for(int i=0;i<num_of_threadblocks;i++)
		{
			if(cpuQPs[i]->cpugpu.produce(img_idx))
			{
				produced=true;
				break;
			}
			else
				checkQueueComplition(num_of_threadblocks,cpuQPs,finished, total_distance );
		}
	}
}
void QueueProduceBlock(int blockId,int num_of_threadblocks,QP **cpuQPs,int img_idx,int * finished, double* total_distance )
{
	bool produced=false;
	while(!produced)
	{
			if(cpuQPs[blockId]->cpugpu.produce(img_idx))
			{
				produced=true;
				break;
			}
			else
				checkQueueComplition(num_of_threadblocks,cpuQPs,finished, total_distance );
	}
}
int main(void) {
	uchar *images1;
	uchar *images2;
	CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
	CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
	load_image_pairs(images1, images2);
	double t_start, t_finish;
	double total_distance=0;



	int num_of_threadblocks = calcNumOfThreadblocks();
	QP **cpuQPs,**gpuQPs;
	CUDA_CHECK( cudaHostAlloc(&cpuQPs, num_of_threadblocks*sizeof(QP*), 0) );
	CUDA_CHECK( cudaHostGetDevicePointer(&gpuQPs,cpuQPs,0) );

	for(int i=0;i<num_of_threadblocks;i++)
	{
		CUDA_CHECK( cudaHostAlloc(&cpuQPs[i], sizeof(QP), 0) );
		cpuQPs[i]->cpugpu=cpu2gpuQueue();
		cpuQPs[i]->gpucpu=gpu2cpuQueue();
		CUDA_CHECK( cudaHostGetDevicePointer(&gpuQPs[i],cpuQPs[i],0) );
	}


	uchar *gpu_image1, *gpu_image2;
	CUDA_CHECK(cudaMalloc(&gpu_image1,SQR(IMG_DIMENSION)*N_IMG_PAIRS));
	CUDA_CHECK(cudaMalloc(&gpu_image2,SQR(IMG_DIMENSION)*N_IMG_PAIRS));
    CUDA_CHECK(cudaMemcpy(gpu_image1,images1, SQR(IMG_DIMENSION)*N_IMG_PAIRS,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gpu_image2,images2, SQR(IMG_DIMENSION)*N_IMG_PAIRS,cudaMemcpyHostToDevice));

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
	test<<<num_of_threadblocks, 1024>>>(gpuQPs,gpu_image1,gpu_image2);
	int finished=0;
	for(int i=0;i<NREQUESTS;i++)
		{
			int img_idx = i % N_IMG_PAIRS;
			checkQueueComplition(num_of_threadblocks,cpuQPs,&finished, &total_distance );
			QueueProduce(num_of_threadblocks,cpuQPs,img_idx,&finished, &total_distance );
		}
	for(int i=0;i<num_of_threadblocks;i++)
		QueueProduceBlock(i,num_of_threadblocks,cpuQPs,-1,&finished, &total_distance );
	while(finished<NREQUESTS)
		checkQueueComplition(num_of_threadblocks,cpuQPs,&finished, &total_distance );
	CUDA_CHECK( cudaDeviceSynchronize());
	printf("average distance between images %f\n", total_distance / NREQUESTS);
    CUDA_CHECK(cudaFree(gpu_image1));
    CUDA_CHECK(cudaFree(gpu_image2));
	for(int i=0;i<num_of_threadblocks;i++)
		CUDA_CHECK( cudaFreeHost(cpuQPs[i]) );
	CUDA_CHECK( cudaFreeHost(cpuQPs) );
	return 0;
}
__global__ void test1(int *gpu)
{
	__shared__ int  h[60000];
	if(!threadIdx.x)printf("gpu\n");
	if(!threadIdx.x)*gpu=1;
	__threadfence_system();
	if(!threadIdx.x)while(*gpu!=2);

	__syncthreads();
}
int main2()
{

	int *cpu,*gpu;
	CUDA_CHECK( cudaHostAlloc(&cpu, sizeof(int), 0) );
	CUDA_CHECK( cudaHostGetDevicePointer(&gpu,cpu,0) );
	*cpu=0;

	test1<<<1, 1024>>>(gpu);
	while(!*cpu){
		cudaError_t ret=cudaStreamQuery(0);
	}
	CUDA_CHECK( cudaDeviceSynchronize());
	printf("cpu=%d\n",*cpu);
	*cpu=2;
	CUDA_CHECK( cudaDeviceSynchronize());
	CUDA_CHECK( cudaFreeHost(cpu) );
	return 1;
}

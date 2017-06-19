/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#define HW2_CU_

#ifdef HW2_CU_

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>


//#include "workelement.h"
#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 10000
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








double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        usleep(1. / (rate_limit->lambda * 1e-6) * 0.01);
    }
}

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

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

__global__ void gpu_image_to_histogram(uchar *image, int *histogram) {
    uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION, threadIdx.x % IMG_DIMENSION);
    atomicAdd(&histogram[pattern], 1);
}

__device__ void gpu_device_image_to_histogram(uchar *image, int *histogram) {
	for(int i=threadIdx.x;i<SQR(IMG_DIMENSION);i+=blockDim.x)
	{
	    uchar pattern = local_binary_pattern(image, i / IMG_DIMENSION, i % IMG_DIMENSION);
	    atomicAdd(&histogram[pattern], 1);
	}
}

__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance) {
    int length = 256;
    int tid = threadIdx.x;
    distance[tid] = 0;
    if (h1[tid] + h2[tid] != 0) {
        distance[tid] = ((double)SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
    }
    h1[tid] = h2[tid]=0;
    __syncthreads();


    while (length > 1) {
        if (threadIdx.x < length / 2) {
            distance[tid] = distance[tid] + distance[tid + length / 2];
        }
        length /= 2;
        __syncthreads();
    }
}

__device__ void gpu_device_histogram_distance(int *h1, int *h2, double *distance) {

	int length=HIST_SIZE;
	for(int i=threadIdx.x;i<HIST_SIZE;i+=blockDim.x)
	{
	    distance[i] = 0;
	    if (h1[i] + h2[i] != 0) {
	        distance[i] = ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
	    }
	    h1[i] = h2[i]=0;
	}
    __syncthreads();

    while (length > 1) {
    	for(int i=threadIdx.x;i<length/2;i+=blockDim.x)
    		distance[i] = distance[i] + distance[i + length / 2];
        length /= 2;
        __syncthreads();
    }
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}





/*************************************************/
/*******CLASS***WORK***ELEMENT********************/
/*************************************************/




class work_element {
public:
	work_element();
	virtual ~work_element();
	bool is_free(){return free;}
	void do_kernel();
	bool check_kernel_finished();
	void update(uchar* cpu_img1,uchar* cpu_img2,double* start,double* finish){cpu_imgs[0]=cpu_img1;cpu_imgs[1]=cpu_img2;time_start=start;time_finish=finish;}
	static double total_distance;
private:
	cudaStream_t stream;
	uchar* cpu_imgs[2];
	uchar* gpu_imgs[2];
	double cpu_distance;
	double * gpu_distance;
	int* gpu_hists[2];
	bool free;
	cudaEvent_t events[2];
	double* time_start;
	double* time_finish;
};

double work_element::total_distance=0;






work_element::work_element():cpu_distance(-1),free(true),time_start(NULL),time_finish(NULL) {
	CUDA_CHECK(cudaStreamCreate(&stream));
	//CUDA_CHECK(cudaEventCreate(&event));
	CUDA_CHECK(cudaMalloc(&gpu_distance,HIST_SIZE*sizeof(double)));
	int i;
	for(i=0;i<2;i++)
	{
		cpu_imgs[i]=NULL;
		CUDA_CHECK(cudaMalloc(&gpu_imgs[i],SQR(IMG_DIMENSION)*sizeof(uchar)));
		CUDA_CHECK(cudaMalloc(&gpu_hists[i],HIST_SIZE*sizeof(int)));
		CUDA_CHECK(cudaEventCreate(&events[i]));
		CUDA_CHECK(cudaMemset(gpu_hists[i], 0, HIST_SIZE * sizeof(int)));

	}

}

work_element::~work_element() {
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFree(gpu_distance));
	int i;
	for(i=0;i<2;i++)
	{
		CUDA_CHECK(cudaFree(gpu_imgs[i]));
		CUDA_CHECK(cudaFree(gpu_hists[i]));
		CUDA_CHECK(cudaEventDestroy(events[i]));
	}

}
void work_element::do_kernel(){
	free=false;
	//*time_finish=get_time_msec();
	//CUDA_CHECK(cudaEventRecord(events[0],stream));
	int i;
	for(i=0;i<2;i++)
	{
		CUDA_CHECK(cudaMemcpyAsync(gpu_imgs[i],cpu_imgs[i],SQR(IMG_DIMENSION)*sizeof(uchar),cudaMemcpyHostToDevice,stream));
		gpu_image_to_histogram <<< 1, 1024 , 0,stream >>> (gpu_imgs[i], gpu_hists[i]);
	}
	gpu_histogram_distance<<<1, 256>>>(gpu_hists[0], gpu_hists[1], gpu_distance);
	CUDA_CHECK(cudaMemcpyAsync(&cpu_distance,gpu_distance,sizeof(double),cudaMemcpyDeviceToHost,stream));
	//CUDA_CHECK(cudaEventRecord(events[1],stream));

}

bool work_element::check_kernel_finished(){
	if(free) return false;
	cudaError_t ret=cudaStreamQuery(stream);
	if(ret==cudaErrorInvalidResourceHandle)
	{
		 printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(ret));
		 exit(1);
	}
	if(ret==cudaErrorNotReady) return false;
	//CUDA_CHECK(cudaEventElapsedTime(&tmp,events[0],events[1]));
	*time_finish=get_time_msec();;
	total_distance+=cpu_distance;
	free=true;
	return true;


}



/*************************************************/
/*************************************************/




void check_completed_no_block(work_element* streams){
	int i;
	for(i=0;i<N_STREMS;i++)
		streams[i].check_kernel_finished();
}
work_element& find_free_stream(work_element* streams)
{
	int i;
	for(i=0;i<N_STREMS;i++)
	if(streams[i].is_free()) return streams[i];
	while(i||!i)
	{
    	for(i=0;i<N_STREMS;i++)
    		if(streams[i].check_kernel_finished()) return streams[i];
	}
	return streams[1]; //not arrive here
}
void check_completed_block(work_element* streams){
	int i,j=0;
	while(j!=N_STREMS)
	{
    	for(i=0,j=0;i<N_STREMS;i++)
    			j+=(streams[i].is_free()||(int)streams[i].check_kernel_finished());
	}
}








#define QUEUE_SIZE 10
class cpu2gpuQueue {
public:
	cpu2gpuQueue():size(QUEUE_SIZE),head(0),tail(0){}
	~cpu2gpuQueue(){}
	__host__ int produce(int img_idx/*,double* finish*/);
	__device__ int consume(int* img_idx/*,double** finish*/);
private:
	volatile int size;
	volatile int head;
	volatile int tail;
	int q[QUEUE_SIZE];
	//double* q_finished_times[QUEUE_SIZE];
};
__device__ int cpu2gpuQueue::consume(int* img_idx/*,double** finish*/)
{
	if(!(tail<head))return 0;
	*img_idx=q[(tail%QUEUE_SIZE)];
	//*finish=q_finished_times[(tail%QUEUE_SIZE)];
	size++;
	tail++;
	__threadfence_system();
	return 1;
}
__host__ int cpu2gpuQueue::produce(int img_idx/*,double* finish*/)
{
	if(!(head<size))return 0;
	q[(head%QUEUE_SIZE)]=img_idx;
	//q_finished_times[(head%QUEUE_SIZE)]=finish;
	head++;
	return 1;
}
class gpu2cpuQueue {
public:
	gpu2cpuQueue():size(QUEUE_SIZE),head(0),tail(0){}
	~gpu2cpuQueue(){}
	__device__ int produce(double distance/*,double* finish*/);
	__host__ int consume(double* distance);
private:
	volatile int size;
	volatile int head;
	volatile int tail;
	double q[QUEUE_SIZE];
	//double* q_finished_times[QUEUE_SIZE];
};
__host__ int gpu2cpuQueue::consume(double* distance)
{
	if(!(tail<head))return 0;
	*distance=q[(tail%QUEUE_SIZE)];
	//double time=get_time_msec();
	//*(q_finished_times[(tail%QUEUE_SIZE)])=get_time_msec();
	//if(!*(q_finished_times[(tail%QUEUE_SIZE)]))printf("error\n");
	size++;
	tail++;
	return 1;
}
__device__ int gpu2cpuQueue::produce(double distance/*,double* finish*/)
{
	if(!(head<size)) return 0;
	q[(head%QUEUE_SIZE)]=distance;
	//q_finished_times[(head%QUEUE_SIZE)]=finish;
	__threadfence_system();
	head++;
	__threadfence_system();
	return 1;
}
typedef struct {
	cpu2gpuQueue cpugpu;
	gpu2cpuQueue gpucpu;
} QP;
void checkQueueComplition(int num_of_threadblocks,QP **cpuQPs,int * finished, double* total_distance,double *req_t_end )
{

	double distance;
	for(int i=0,ret;i<num_of_threadblocks;i++)
	{
		do{
			ret=cpuQPs[i]->gpucpu.consume(&distance);
			*total_distance+=ret*distance;
			if(ret) req_t_end[*finished]=get_time_msec();
			*finished+=ret;
		}while(ret);
	}
}
void QueueProduce(int num_of_threadblocks,QP **cpuQPs,int img_idx,int * finished, double* total_distance,double *req_t_end  )
{
	bool produced=false;
	while(!produced)
	{
		for(int i=0;i<num_of_threadblocks;i++)
		{
			if(cpuQPs[i]->cpugpu.produce(img_idx/*,finish_time*/))
			{
				produced=true;
				break;
			}
			else
				checkQueueComplition(num_of_threadblocks,cpuQPs,finished, total_distance,req_t_end );
		}
	}
}
void QueueProduceBlock(int blockId,int num_of_threadblocks,QP **cpuQPs,int img_idx,int * finished, double* total_distance,double *req_t_end )
{
	bool produced=false;
	while(!produced)
	{
			if(cpuQPs[blockId]->cpugpu.produce(img_idx/*,NULL*/))
			{
				produced=true;
				break;
			}
			else
				checkQueueComplition(num_of_threadblocks,cpuQPs,finished, total_distance ,req_t_end);
	}
}

__global__ void kernel_queue_mode(QP** gpuQPs,uchar* imags1,uchar* imags2 ){
	__shared__ int hist1[HIST_SIZE],hist2[HIST_SIZE];
	__shared__ double distance[HIST_SIZE];
	__shared__ int img_idx;




	for(int i=threadIdx.x;i<HIST_SIZE;i+=blockDim.x)
		hist1[i]=hist2[i]=0;

	bool running=true;
	//for(int i=0;i<NREQUESTS;i++)
	while( running )
	{
		if(!threadIdx.x)while(!gpuQPs[blockIdx.x]->cpugpu.consume(&img_idx/*,&finish_time*/));
		__syncthreads();
		//printf("img_idx=%d\n",img_idx);
		if(img_idx==-1)
			break;
		gpu_device_image_to_histogram(&imags1[img_idx*SQR(IMG_DIMENSION)],hist1);
		gpu_device_image_to_histogram(&imags2[img_idx*SQR(IMG_DIMENSION)],hist2);
		__syncthreads();
		gpu_device_histogram_distance(hist1,hist2,distance);
		__syncthreads();
		//if(!threadIdx.x)total_distance+=distance[0];
		if(!threadIdx.x)while(!gpuQPs[blockIdx.x]->gpucpu.produce(distance[0]/*,finish_time*/));
		__syncthreads();
	}
	__syncthreads();
	//if(!threadIdx.x)printf("gpu average distance between images %f\n", total_distance / NREQUESTS);
}




















int calcNumOfThreadblocks(int threadsPerBlock){//TODO: implement
	/*
	 * amount of shared momory per block:
	 * 	__shared__ int hist1[HIST_SIZE],hist2[HIST_SIZE];
	__shared__ double distance[HIST_SIZE];
	__shared__ double total_distance;
	__shared__ int img_idx;
	 */
	int sharedMomoryPerBlock=(2*HIST_SIZE+1)*(sizeof(int)+sizeof(double));
	int regsPerThread = 32;
	int regsPerBlock=regsPerThread*threadsPerBlock;
	int ret = 0;
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
	   cudaDeviceProp prop;
	   cudaGetDeviceProperties(&prop, i);
	   int sm = prop.multiProcessorCount;
	   int maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
	   size_t sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
	   int regsPerMultiprocessor = prop.regsPerMultiprocessor;

	   int numOfBlocksPerSmSharedMem=sharedMemPerMultiprocessor/sharedMomoryPerBlock;
	   int numOfBlocksPerSmRegs=regsPerMultiprocessor/regsPerBlock;
	   int numOfBlocksPerSmThreads=maxThreadsPerMultiProcessor/threadsPerBlock;

	   //find the minimum of blocks per sm
	   int minBlocksPerSm=(numOfBlocksPerSmSharedMem<numOfBlocksPerSmRegs)?numOfBlocksPerSmSharedMem:numOfBlocksPerSmRegs;
	   minBlocksPerSm=(numOfBlocksPerSmThreads<minBlocksPerSm)?numOfBlocksPerSmThreads:minBlocksPerSm;

	   //find the minimum of blocks for the device
	   int minBlocksForDevice=minBlocksPerSm*sm;
	   ret += minBlocksForDevice;
	}
	return ret;

}

void run_queue(double cpu_distance, double* avg_latency,double* throughput,uchar*images1,uchar*images2,int threads_queue_mode,\
		double load)
{

	double total_distance = 0;
	double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
	memset(req_t_start, 0, NREQUESTS * sizeof(double));

	double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
	memset(req_t_end, 0, NREQUESTS * sizeof(double));

	struct rate_limit_t rate_limit;
	rate_limit_init(&rate_limit, load, 0);
	work_element streams[N_STREMS];
	double ti = get_time_msec();

	int num_of_threadblocks = calcNumOfThreadblocks(threads_queue_mode); //TODO
	int finished=0;
	uchar *gpu_image1, *gpu_image2;
    CUDA_CHECK(cudaMalloc(&gpu_image1,SQR(IMG_DIMENSION)*N_IMG_PAIRS));
	CUDA_CHECK(cudaMalloc(&gpu_image2,SQR(IMG_DIMENSION)*N_IMG_PAIRS));
	CUDA_CHECK(cudaMemcpy(gpu_image1,images1, SQR(IMG_DIMENSION)*N_IMG_PAIRS,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gpu_image2,images2, SQR(IMG_DIMENSION)*N_IMG_PAIRS,cudaMemcpyHostToDevice));

	QP **cpuQPs,**gpuQPs;
	CUDA_CHECK( cudaHostAlloc(&cpuQPs, num_of_threadblocks*sizeof(QP*), 0) );
	CUDA_CHECK( cudaHostGetDevicePointer( &gpuQPs,cpuQPs ,0 ) );

	for(int i=0;i<num_of_threadblocks;i++)
	{
        CUDA_CHECK( cudaHostAlloc(&cpuQPs[i], sizeof(QP), 0) );
	    cpuQPs[i]->cpugpu=cpu2gpuQueue();
	    cpuQPs[i]->gpucpu=gpu2cpuQueue();
	    CUDA_CHECK( cudaHostGetDevicePointer(&gpuQPs[i],cpuQPs[i],0) );
	 }
	kernel_queue_mode<<<num_of_threadblocks, threads_queue_mode>>>(gpuQPs,gpu_image1,gpu_image2);

	for(int i=0;i<NREQUESTS;i++)
	{
		int img_idx = i % N_IMG_PAIRS;
		checkQueueComplition(num_of_threadblocks,cpuQPs,&finished, &total_distance,req_t_end );
		req_t_end[i] = get_time_msec();
		QueueProduce(num_of_threadblocks,cpuQPs,img_idx,&finished, &total_distance,req_t_end);
	}
	for(int i=0;i<num_of_threadblocks;i++)
		QueueProduceBlock(i,num_of_threadblocks,cpuQPs,-1,&finished, &total_distance,req_t_end );
	while(finished<NREQUESTS)
		checkQueueComplition(num_of_threadblocks,cpuQPs,&finished, &total_distance,req_t_end );
	CUDA_CHECK( cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(gpu_image1));
	CUDA_CHECK(cudaFree(gpu_image2));
	for(int i=0;i<num_of_threadblocks;i++)
		CUDA_CHECK( cudaFreeHost(cpuQPs[i]) );
	CUDA_CHECK( cudaFreeHost(cpuQPs) );
	double tf = get_time_msec();
	*avg_latency=0;
	for (int i = 0; i < NREQUESTS; i++) {
		*avg_latency += (req_t_end[i] - req_t_start[i]);
	}
	*avg_latency /= NREQUESTS;
	* throughput=NREQUESTS / (tf - ti) * 1e+3;

	if(total_distance/NREQUESTS-cpu_distance>0.000001||total_distance-cpu_distance<-0.000001) printf("baaaaad\n");
	free(req_t_start);
	free(req_t_end);
	return;




}



void run_queue_avg(double cpu_distance, double* avg_latency,double* throughput,uchar*images1,uchar*images2,int threads_queue_mode,\
		double load)
{
	double throughput_sum=0 , avg_latency_sum = 0;
	for(int i=0;i<100;i++)
	{
		run_queue(cpu_distance, avg_latency,throughput,images1,images2,threads_queue_mode,load);
		throughput_sum+=(*throughput);
		avg_latency_sum+=(*avg_latency);
	}
	*throughput=throughput_sum/100;
	*avg_latency=avg_latency_sum/100;
    printf("load = %lf (req/sec)\n", load);
    printf("throughput = %lf (req/sec)\n",* throughput);
    printf("average latency = %lf (msec)\n", *avg_latency);
    return;
}

enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if        (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );

    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance;

    /* using CPU */
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


    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");
    do {
        uchar *gpu_image1, *gpu_image2;
        int *gpu_hist1, *gpu_hist2;
        double *gpu_hist_distance;
        double cpu_hist_distance;
        cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION);
        cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION);
        cudaMalloc(&gpu_hist1, 256 * sizeof(int));
        cudaMalloc(&gpu_hist2, 256 * sizeof(int));
        cudaMalloc(&gpu_hist_distance, 256 * sizeof(double));

        total_distance = 0;
        t_start = get_time_msec();
        for (int i = 0; i < NREQUESTS; i++) {
            int img_idx = i % N_IMG_PAIRS;
            cudaMemcpy(gpu_image1, &images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_image2, &images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
            cudaMemset(gpu_hist1, 0, 256 * sizeof(int));
            cudaMemset(gpu_hist2, 0, 256 * sizeof(int));
            gpu_image_to_histogram<<<1, 1024>>>(gpu_image1, gpu_hist1);
            gpu_image_to_histogram<<<1, 1024>>>(gpu_image2, gpu_hist2);
            gpu_histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
            cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost);
            total_distance += cpu_hist_distance;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("average distance between images %f\n", total_distance / NREQUESTS);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);
        cudaFree(gpu_image1);
        cudaFree(gpu_image2);
        cudaFree(gpu_hist1);
        cudaFree(gpu_hist2);
        cudaFree(gpu_hist_distance);

    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    total_distance = 0;
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));


    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);
    work_element streams[N_STREMS];



    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
        for (int i = 0; i < NREQUESTS; i++) {
        	check_completed_no_block(streams);
            rate_limit_wait(&rate_limit);
            req_t_start[i]=get_time_msec();

            int img_idx = i % N_IMG_PAIRS;
            work_element& free_stream=find_free_stream(streams);
            free_stream.update(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION],&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION],&req_t_start[i],&req_t_end[i]);
            free_stream.do_kernel();
        }
        check_completed_block(streams);
        total_distance=work_element::total_distance;



    } else if (mode == PROGRAM_MODE_QUEUE) {
    	//calc num of thread blocks that can currently run in the GPU
    	int num_of_threadblocks = calcNumOfThreadblocks(threads_queue_mode); //TODO
    	if(!num_of_threadblocks)
    	{
    		printf("the number of thread blocks that can run is 0!!\n");
    		exit(1);
    	}
    	/*double avg_latency_tmp,throughput,throughput_sum=0,avg_latency=0;
    	run_queue_avg(cpu_distance,&avg_latency,&throughput,images1,images2,threads_queue_mode,load);

    	double max_load=throughput;
    	for(int k=0;k<10;k++)
    	{
    		load=(max_load/10)+(max_load*2-max_load/10)*k/10;
    		run_queue_avg(cpu_distance, &avg_latency_tmp,&throughput,images1,images2,threads_queue_mode,load);
    	}*/

    	//run_queue(cpu_distance, &avg_latency,&throughput,images1,images2,threads_queue_mode,load);
    	int finished=0;
    	uchar *gpu_image1, *gpu_image2;
    	CUDA_CHECK(cudaMalloc(&gpu_image1,SQR(IMG_DIMENSION)*N_IMG_PAIRS));
    	CUDA_CHECK(cudaMalloc(&gpu_image2,SQR(IMG_DIMENSION)*N_IMG_PAIRS));
        CUDA_CHECK(cudaMemcpy(gpu_image1,images1, SQR(IMG_DIMENSION)*N_IMG_PAIRS,cudaMemcpyHostToDevice));
    	CUDA_CHECK(cudaMemcpy(gpu_image2,images2, SQR(IMG_DIMENSION)*N_IMG_PAIRS,cudaMemcpyHostToDevice));



    	QP **cpuQPs,**gpuQPs;
    	CUDA_CHECK( cudaHostAlloc(&cpuQPs, num_of_threadblocks*sizeof(QP*), 0) );
    	CUDA_CHECK( cudaHostGetDevicePointer( &gpuQPs,cpuQPs ,0 ) );
    	//return 0;

    	for(int i=0;i<num_of_threadblocks;i++)
    	{
    		CUDA_CHECK( cudaHostAlloc(&cpuQPs[i], sizeof(QP), 0) );
    		cpuQPs[i]->cpugpu=cpu2gpuQueue();
    		cpuQPs[i]->gpucpu=gpu2cpuQueue();
    		CUDA_CHECK( cudaHostGetDevicePointer(&gpuQPs[i],cpuQPs[i],0) );
    	}
    	kernel_queue_mode<<<num_of_threadblocks, threads_queue_mode>>>(gpuQPs,gpu_image1,gpu_image2);

    	for(int i=0;i<NREQUESTS;i++)
    		{
    			int img_idx = i % N_IMG_PAIRS;
    			checkQueueComplition(num_of_threadblocks,cpuQPs,&finished, &total_distance,req_t_end );
    			rate_limit_wait(&rate_limit);
    			req_t_start[i] = get_time_msec();
    			//req_t_end[i] = get_time_msec();
    			QueueProduce(num_of_threadblocks,cpuQPs,img_idx,&finished, &total_distance,req_t_end);
    		}
   		for(int i=0;i<num_of_threadblocks;i++)
   			QueueProduceBlock(i,num_of_threadblocks,cpuQPs,-1,&finished, &total_distance,req_t_end );
   		while(finished<NREQUESTS)
   			checkQueueComplition(num_of_threadblocks,cpuQPs,&finished, &total_distance,req_t_end );
   		CUDA_CHECK( cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(gpu_image1));
   	    CUDA_CHECK(cudaFree(gpu_image2));
   		for(int i=0;i<num_of_threadblocks;i++)
   			CUDA_CHECK( cudaFreeHost(cpuQPs[i]) );
   		CUDA_CHECK( cudaFreeHost(cpuQPs) );


    } else {
        assert(0);
    }
    double tf = get_time_msec();

    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
        if((req_t_end[i] - req_t_start[i])<0)
        {
        	//avg_latency -= (req_t_end[i] - req_t_start[i]);
        	//printf("error,i=%d\n",i);
        }

    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    CUDA_CHECK( cudaFreeHost(images1) );
    CUDA_CHECK( cudaFreeHost(images2) );
    free(req_t_start);
    free(req_t_end);
    return 0;
}
#endif

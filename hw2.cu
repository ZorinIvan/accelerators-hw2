/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>


#include "workelement.h"
#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 100000
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
	*time_start=get_time_msec();
	CUDA_CHECK(cudaEventRecord(events[0],stream));
	int i;
	for(i=0;i<2;i++)
	{
		CUDA_CHECK(cudaMemcpyAsync(gpu_imgs[i],cpu_imgs[i],SQR(IMG_DIMENSION)*sizeof(uchar),cudaMemcpyHostToDevice,stream));
		gpu_image_to_histogram <<< 1, 1024 , 0,stream >>> (gpu_imgs[i], gpu_hists[i]);
	}
	gpu_histogram_distance<<<1, 256>>>(gpu_hists[0], gpu_hists[1], gpu_distance);
	CUDA_CHECK(cudaMemcpyAsync(&cpu_distance,gpu_distance,sizeof(double),cudaMemcpyDeviceToHost,stream));
	CUDA_CHECK(cudaEventRecord(events[1],stream));

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
	float tmp=0;
	CUDA_CHECK(cudaEventElapsedTime(&tmp,events[0],events[1]));
	*time_finish=*time_start+tmp;
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
        uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
        int *gpu_hist1, *gpu_hist2; // TODO: allocate with cudaMalloc
        double *gpu_hist_distance; //TODO: allocate with cudaMalloc
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

    /* TODO allocate / initialize memory, streams, etc... */
    //cudaEvent_t
    work_element streams[N_STREMS];



    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
        for (int i = 0; i < NREQUESTS; i++) {

            /* TODO query (don't block) streams for any completed requests.
               update req_t_end of completed requests
               update total_distance */
        	check_completed_no_block(streams);//TODO impl
            rate_limit_wait(&rate_limit);
            int img_idx = i % N_IMG_PAIRS;
            work_element& free_stream=find_free_stream(streams);
            free_stream.update(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION],&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION],&req_t_start[i],&req_t_end[i]);
            free_stream.do_kernel();
        }
        check_completed_block(streams);
        total_distance=work_element::total_distance;


    } else if (mode == PROGRAM_MODE_QUEUE) {
        for (int i = 0; i < NREQUESTS; i++) {

            /* TODO check producer consumer queue for any responses.
               don't block. if no responses are there we'll check again in the next iteration
               update req_t_end of completed requests 
               update total_distance */

            rate_limit_wait(&rate_limit);
            int img_idx = i % N_IMG_PAIRS;
            req_t_start[i] = get_time_msec();

            /* TODO place memcpy's and kernels in a stream */
        }
        /* TODO wait until you have responses for all requests */
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    return 0;
}

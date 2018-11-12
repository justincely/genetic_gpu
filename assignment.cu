#include <stdio.h>
#include <iostream>
#include <math.h>
/* thrust algorithm */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

const float MAX = 5.12;
const float MIN = -5.12;
const int n_population = 1000;
const int n_parameters = 10;

/**
    Convolves an array with a window, using shared memory for caching.

    @param n the size of the array.
    @param *a the pointer to the array being convolved.
    @param size the size of the convolution window.
    @param iterations the number of convolutions to go through.
    @param *w the pointer to the convolution window.
 */
__global__ void generateIndividuals(float a[n_population][n_parameters], float min, float max) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        printf("Hello from block %d, thread %d. Current total index %d, using stride size %d\n", blockIdx.x, threadIdx.x, index, stride);
        /**
           for (int i=index; i<n_population; i+=stride) {
           for (int p=0; p<n_parameters; p++) {

            float myrandf = curand_uniform(randState+i);
            //myrandf *= (max_rand_int[index] - min_rand_int[index]+0.999999);
            //myrandf += min_rand_int[index];
            printf("%f\n", myrandf);
            a[i] = myrandf;
            //a[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max));
           }
           }
         */
}


struct prg
{
        float a, b;

        __host__ __device__
        prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {
        };

        __host__ __device__
        float operator()(const unsigned int n) const
        {
                thrust::default_random_engine rng;
                thrust::uniform_real_distribution<float> dist(a, b);
                rng.discard(n);

                return dist(rng);
        }
};

struct normal
{
        float a, b;

        __host__ __device__
        normal(float _a=0.f, float _b=0.1f) : a(_a), b(_b) {
        };

        __host__ __device__
        float operator()(const unsigned int n) const
        {
                thrust::default_random_engine rng;
                thrust::normal_distribution<float> dist(a, b);
                rng.discard(n);

                return dist(rng);
        }
};


__host__ static __inline__ float randFloat() {
        return MIN + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(MAX-MIN)));
}

__host__ static __inline__ float randInt()
{
        return ((int)rand());
}


//__device__ evaluate()


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
        __host__ __device__
        T operator()(const T& x) const {
                return x * x;
        }
};

//
template <typename T>
struct minus5
{
        __host__ __device__
        T operator()(const T& x) const {
                return x - 0.5;
        }
};


__host__ float evaluateMember(int index, float *array) {
        float sum = 0;
        for (int i=0; i<n_parameters; i++) {
                sum += pow(0.5 - array[index*n_parameters+i], 2);
        }
        return sqrt(sum);
}


__host__ thrust::device_vector<float> evaluateGeneration(thrust::device_vector<float> population){
  thrust::device_vector<float> popScores(n_population);
  thrust::fill(popScores.begin(), popScores.end(), 0.0);
  
  thrust::device_vector<float> temp(population.size());
  thrust::copy(thrust::device, population.begin(), population.end(), temp.begin());

  thrust::transform(temp.begin(), temp.end(), temp.begin(), minus5<float>());
  thrust::transform(temp.begin(), temp.end(), temp.begin(), square<float>());

  for (int i=0; i<n_population; i++) {
          //std::cout << "Individual " << i << ":  ";
          //for (int offset=0; offset<n_parameters; offset++) {
          //        std::cout << population[i*n_parameters+offset] << " ";
          //}

          float score = std::sqrt(thrust::reduce(temp.begin()+i*n_parameters, temp.begin()+i*n_parameters+n_parameters, (float) 0, thrust::plus<float>()));
          popScores[i] = (float) 1.0 / (float) (score+1.0);

          //std::cout << " - Score: " << score << endl;
  }
  
  return popScores;
}

int main(int argc, char** argv) {
        /* initialize random seed for timing purposes */
        //srand (time(NULL));
        srand (static_cast <unsigned> (time(0)));

        // Cude Device Properties to see if the blocksize can be handled
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // Generate initial random population
        thrust::device_vector<float> population(n_population * n_parameters);
        thrust::counting_iterator<unsigned int> index_sequence_begin(0);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + n_population * n_parameters,
                          population.begin(),
                          prg(-5.12f,5.12f));

        int generation = 0;
        while (generation < 20) {
              thrust::device_vector<float> popScores = evaluateGeneration(population);
              float best = *(thrust::min_element(popScores.begin(), popScores.end()));

              generation++;
              std::cout << "Bred generation " << generation << " Best score: " << best << endl;
        }



        return 0;
}

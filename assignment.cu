#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>

/* thrust algorithm */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

/* parse cmdline options */
#include <boost/program_options.hpp>
#include <exception>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
using namespace std;
namespace po = boost::program_options;

const double MAX = 5.12;
const double MIN = -5.12;
int n_population = 200;
int n_parameters = 10;


struct prg
{
        double a, b;

        __host__ __device__
        prg(double _a=0.f, double _b=1.f) : a(_a), b(_b) {
        };

        __host__ __device__
        double operator()(const unsigned int n) const
        {
                thrust::default_random_engine rng;
                //thrust::default_random_engine rng( 5555555 );
                thrust::uniform_real_distribution<double> dist(a, b);
                rng.discard(n);

                return dist(rng);
        }
};

struct normal
{
        double a, b;

        __host__ __device__
        normal(double _a=0.f, double _b=0.1f) : a(_a), b(_b) {
        };

        __host__ __device__
        double operator()(const unsigned int n) const
        {
                thrust::default_random_engine rng;
                thrust::normal_distribution<double> dist(a, b);
                rng.discard(n);

                return dist(rng);
        }
};

// return the biggest of two tuples
template <class T>
struct larger_tuple {
        __device__ __host__
        thrust::tuple<T,int> operator()(const thrust::tuple<T,int> &a, const thrust::tuple<T,int> &b)
        {
                if (a > b) return a;
                else return b;
        }

};

template <class T>
int min_index(thrust::device_vector<T>& vec) {

        // create implicit index sequence [0, 1, 2, ... )
        thrust::counting_iterator<int> begin(0); thrust::counting_iterator<int> end(vec.size());
        thrust::tuple<T,int> init(vec[0],0);
        thrust::tuple<T,int> largest;

        largest = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(vec.begin(), begin)), thrust::make_zip_iterator(thrust::make_tuple(vec.end(), end)),
                                 init, larger_tuple<T>());
        return get<1>(largest);
}


__global__ void score(unsigned int n, unsigned int np, double *source, double *score) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        double value;

        if (index < n) {
                for (int i=index; i < n; i += stride) {
                        value = 0;
                        for (int p=0; p<np; p++) {
                                value += std::pow(source[i*np+p]-0.5, 2);
                        }

                        value = (double) std::sqrt( (double) value);
                        score[i] = (double) 1.0 / (double) (value+1.0);
                }
        }
}

__global__ void pickParents(unsigned int n, unsigned int np, int *randParents, double *score, int *pool) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i=index; i<n; i+=stride) {
                double best = -1.0;
                int best_index = -1;
                int idx;
                for (int j=0; j<7; j++) {
                        idx = randParents[i*7+j];
                        if (score[idx] > best) {
                                best = score[idx];
                                best_index = idx;
                        }
                }
                pool[i] = best_index;
        }
}


__global__ void breedGeneration(unsigned int n, unsigned int np, int *randomParameters, double *population, double *newPopulation, int *parentsPool, double *mutations) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int parentA;
        int parentB;
        int probCopyParents;
        int crossover;
        int probChildAMutate;
        int mutationPoint;

        for (int i=index; i<n; i+=stride) {
                probCopyParents = randomParameters[i*4];
                crossover = randomParameters[(i+1)*4] % np;
                probChildAMutate = randomParameters[(i+2)*4];
                mutationPoint = randomParameters[(i+3)*4] % np;
                parentA = parentsPool[i];
                parentB = parentsPool[i+1];

                //printf("index: %d, parentA: %d, parentB: %d, Copy Parents: %d\n", index, parentA, parentB, probCopyParents);
                //printf("index: %d, crossover: %d, mutateA: %d, mutateB: %d\n", index, crossover, probChildAMutate, probChildBMutate);

                if (probCopyParents < 10) {
                        for (int j=0; j<np; j++) {
                                newPopulation[(i*np) + j] = population[(parentA*np) + j];
                                //newPopulation[((i+1)*n_parameters) + j] = population[(parentB*n_parameters) + j];
                        }
                }

                for (int j=0; j<np; j++) {
                        if (j < crossover) {
                                newPopulation[(i*np) + j] = population[(parentA*np) + j];
                                //newPopulation[((i+1)*n_parameters) + j] = population[(parentB*n_parameters) + j];
                        } else {
                                newPopulation[(i*np) + j] = population[(parentB*np) + j];
                                //newPopulation[((i+1)*n_parameters) + j] = population[(parentA*n_parameters) + j];
                        }
                }

                if (probChildAMutate < 5) {
                        double newval = newPopulation[(i*np) + mutationPoint] + mutations[i];
                        newPopulation[(i*np) + mutationPoint] = fminf(newval, MAX);
                        newPopulation[(i*np) + mutationPoint] = fmaxf(newval, MIN);
                }

        }

}

__host__ void printMember(thrust::device_vector<double> member){
        cout << "Member: ";
        for (int i=0; i<n_parameters; i++) {
                cout << member[i] << " ";
        }
        cout << endl;
}


/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

        int idx = threadIdx.x+blockDim.x*blockIdx.x;

        /* we have to initialize the state */
        curand_init(seed, idx, 0,  &states[idx]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void setRandom(curandState_t* states, int* numbers, int n_population) {
        int idx = threadIdx.x+blockDim.x*blockIdx.x;

        /* curand works like rand - except that it takes a state as a parameter */
        for (int i=0; i<7; i++) {
                numbers[idx+i] = curand(&states[idx]) % n_population;
        }
}


/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void setRandomPercent(curandState_t* states, int* numbers) {
        int idx = threadIdx.x+blockDim.x*blockIdx.x;

        /* curand works like rand - except that it takes a state as a parameter */
        for (int i=0; i<7; i++) {
                numbers[idx+i] = curand(&states[idx]) % 100;
        }
}

// We need this function to define how to sort
// the vector. We will pass this function into the
// third parameter and it will tell it to sort descendingly.
bool reverseSort(double i, double j) {
        return i > j;
}


int main(int argc, char** argv) {
        // Parse Argument variables
        if (argc >= 2) {
                n_population = atoi(argv[1]);
        }
        if (argc >= 3) {
                n_parameters = atoi(argv[2]);
        }


        /* initialize random seed for timing purposes */
        //srand (time(NULL));
        srand (static_cast <unsigned> (time(0)));
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, .1);

        // Cude Device Properties to see if the blocksize can be handled
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // Generate initial random population
        thrust::device_vector<double> population(n_population * n_parameters);
        thrust::counting_iterator<unsigned int> index_sequence_begin(0);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + n_population * n_parameters,
                          population.begin(),
                          prg(MIN, MAX));


        int generation = 0;

        thrust::device_vector<double> popScores(n_population);
        thrust::device_vector<double> newPopulation(n_population * n_parameters);

        double* popPtr = thrust::raw_pointer_cast(&population[0]);
        double* newPopPtr = thrust::raw_pointer_cast(&newPopulation[0]);
        double* scoresPtr = thrust::raw_pointer_cast(&popScores[0]);
        score<<<2048, 1024>>>(n_population, n_parameters, popPtr, scoresPtr);

        double best = *(thrust::max_element(popScores.begin(), popScores.end()));
        int best_index = min_index(popScores);

        std::cout << "Initial generation " << generation << " Best score: " << best << " at index: " << best_index << "   ";
        for (int i=0; i<n_parameters; i++) {
                std::cout << population[best_index * n_parameters + i] << " ";
        }
        std::cout << endl;

        // Create random states and initialize
        curandState_t* states;
        cudaMalloc((void**) &states, n_population*7*sizeof(curandState_t));

        int *randParents;
        cudaMalloc((void**)&randParents, n_population*7*sizeof(int));

        int *parentsPool_d;
        cudaMalloc((void**)&parentsPool_d, n_population*sizeof(int));

        // Create random states and initialize
        curandState_t* childStates;
        cudaMalloc((void**) &childStates, n_population*4*sizeof(curandState_t));

        // Setup device memory and generate random numbers
        int *randParams_d;
        cudaMalloc((void**)&randParams_d, n_population*4*sizeof(int));

        while (best < .99) {
        //while (generation < 2000) {
                init<<<n_population*7, 1>>>(time(0), states);
                // Setup device memory and generate random numbers

                setRandom<<<n_population*7, 1>>>(states, randParents, n_population);

                double* scorePtr = thrust::raw_pointer_cast(&popScores[0]);
                pickParents<<<2048, 1024>>>(n_population, n_parameters, randParents, scorePtr, parentsPool_d);

                init<<<n_population*4, 1>>>(time(0), childStates);

                setRandomPercent<<<n_population*4, 1>>>(childStates, randParams_d);

                // Generate initial random population
                thrust::device_vector<double> mutations(n_population);
                thrust::counting_iterator<unsigned int> index_sequence_begin(0);
                thrust::transform(index_sequence_begin,
                                  index_sequence_begin + n_population,
                                  mutations.begin(),
                                  normal(0.0, 0.001));
                double* mutPtr = thrust::raw_pointer_cast(&mutations[0]);

                breedGeneration<<<2048, 1024>>>(n_population, n_parameters, randParams_d, popPtr, newPopPtr, parentsPool_d, mutPtr);

                thrust::copy(thrust::device, newPopulation.begin(), newPopulation.end(), population.begin());

                score<<<2048, 1024>>>(n_population, n_parameters, popPtr, scoresPtr);
                best = *(thrust::min_element(popScores.begin(), popScores.end()));
                best_index = min_index(popScores);

                std::cout << "Bred generation " << generation << " Best score: " << best << " at index: " << best_index << "   ";
                for (int i=0; i<n_parameters; i++) {
                        std::cout << population[best_index * n_parameters + i] << " ";
                }
                std::cout << endl;

                generation++;
        }

        return 0;
}

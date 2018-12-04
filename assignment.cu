// standard lib stuff
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>

// thrust library
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

// For random number generation
#include <curand.h>
#include <curand_kernel.h>

// Program global variables.  Many will be overridden during execution.
const double MAX = 5.12;
const double MIN = -5.12;
const int SIZE_PARENT_POOL = 7;

// Default settings if not overriden at runtime.
int POPULATION_SIZE = 200;
int N_PARAMETERS = 10;
int BLOCKSIZE = 256;
int TOTALTHREADS = 1024;

/* wrapper function to check cuda calls
 * ref: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */ 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Generate uniform distribution 
 *
 * Used by thrust transform functions to create large numbers of 
 * random numbers in a uniform distribution.
 */
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

/* Generate normal distribution 
 *
 * Used by thrust transform functions to create large numbers of 
 * random numbers from a gaussian (normal) distribution.
 */
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

/* Return the larger of the two elements in the tuple
 *
 * Used by thrust reduce functions to find the maximum element 
 * in an array.
 */
template <class T>
struct larger_tuple {
        __device__ __host__
        thrust::tuple<T,int> operator()(const thrust::tuple<T,int> &a, const thrust::tuple<T,int> &b)
        {
                if (a > b) return a;
                else return b;
        }

};

/* min_index returns the index of the smallest element in the array
 *
 * Similar to NumPy argmin() functionality, this returns the index of the 
 * smallest element in the input array.
 */
template <class T>
int min_index(thrust::device_vector<T>& vec) {
        thrust::counting_iterator<int> begin(0); thrust::counting_iterator<int> end(vec.size());
        thrust::tuple<T,int> init(vec[0],0);
        thrust::tuple<T,int> largest;

        largest = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(vec.begin(), begin)), thrust::make_zip_iterator(thrust::make_tuple(vec.end(), end)),
                                 init, larger_tuple<T>());
        return thrust::get<1>(largest);
}


/* score evaluates the fitness of each member in the population
 *
 * Each member (represented by a N_PARAMETERS sized section of the population 
 * array) is evaulated against the desired input function.  For this example, 
 * this is hard-coded to be the "offset sphere problem". 
 *
 * Higher numbers represent better fitness.
 */
__global__ void score(unsigned int n, unsigned int np, double *source, double *score) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        double value;

        if (index < n) {
                for (int i=index; i < n; i += stride) {
                        // score = 1/ (sqrt(sum((xi - 0.5)**2)) + 1)
                        value = 0;
                        for (int p=0; p<np; p++) {
                                value += std::pow(source[i*np+p]-0.5, 2);
                        }

                        value = (double) std::sqrt( (double) value);
                        score[i] = (double) 1.0 / (double) (value+1.0);
                }
        }
}

/* pickParents generates the set of parents to breed into the next generation
 *
 * The method here is a sort of limited tournament style.  Each parent
 * in the output array is the member with the best fitness drawn from a random
 * pool of SIZE_PARENT_POOL that has been generated ahead of time.  
 * 
 * The brings in a controlled amount of natural selction, while still tending
 * to bring the best members forward to the next generation.  Note many parents
 * can and will breed multiple times. 
 */
__global__ void pickParents(unsigned int n, unsigned int np, int *randParents, double *score, int *pool) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        // Pick a parent n times
        for (int i=index; i<n; i+=stride) {
                double best = -1.0;
                int best_index = -1;
                int idx;
                
                // Grab the SIZE_PARENT_POOL randomely chosen individuals, 
                // and pick the best one to output.
                for (int j=0; j<SIZE_PARENT_POOL; j++) {
                        idx = randParents[i*SIZE_PARENT_POOL+j];
                        if (score[idx] > best) {
                                best = score[idx];
                                best_index = idx;
                        }
                }
                pool[i] = best_index;
        }
}


/* breedGeneration uses the chosen parents to create a new derived generation
 *
 * 10% of the time, the parent will produce no children, but will move straight
 * into the new generation.
 *
 * 90% of the time, a child will be produced from the two parents.  In this 
 * case, half the values from one parent and half from the other are used
 * to create the child. In this event, 5% of the time a random mutation 
 * will also happen to the child's values.
 */
__global__ void breedGeneration(unsigned int n, unsigned int np, int *randomParameters, double *population, double *newPopulation, int *parentsPool, double *mutations) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int parentA;
        int parentB;
        int probCopyParents;
        int crossover;
        int probChildAMutate;
        int mutationPoint;

        // Once for each individual
        for (int i=index; i<n; i+=stride) {
                probCopyParents = randomParameters[i*4];
                crossover = randomParameters[(i+1)*4] % np;
                probChildAMutate = randomParameters[(i+2)*4];
                mutationPoint = randomParameters[(i+3)*4] % np;
                parentA = parentsPool[i];
                parentB = parentsPool[i+1];

                // Place parent directly into the subsequent generation and 
                if (probCopyParents < 10) {
                        for (int j=0; j<np; j++) {
                                newPopulation[(i*np) + j] = population[(parentA*np) + j];
                                continue;
                        }
                }

                for (int j=0; j<np; j++) {
                        if (j < crossover) {
                                newPopulation[(i*np) + j] = population[(parentA*np) + j];
                        } else {
                                newPopulation[(i*np) + j] = population[(parentB*np) + j];
                        }
                }

                if (probChildAMutate < 5) {
                        double newval = newPopulation[(i*np) + mutationPoint] + mutations[i];
                        newPopulation[(i*np) + mutationPoint] = fminf(newval, MAX);
                        newPopulation[(i*np) + mutationPoint] = fmaxf(newval, MIN);
                }

        }

}


/* init sets up the random states
 */
__global__ void init(unsigned int seed, curandState_t* states) {
        int idx = threadIdx.x+blockDim.x*blockIdx.x;

        curand_init(seed, idx, 0,  &states[idx]);
}

/* setRandom generates a random array modulo the max parameter.
 */
__global__ void setRandom(curandState_t* states, int* numbers, int max) {
        int idx = threadIdx.x+blockDim.x*blockIdx.x;

        for (int i=0; i<SIZE_PARENT_POOL; i++) {
                numbers[idx+i] = curand(&states[idx]) % max;
        }
}

int main(int argc, char** argv) {
        int generation = 0;

        // Create thrust arrays to hold the population, scores, and the new 
        // population that is created each generation.
        thrust::device_vector<double> population(POPULATION_SIZE * N_PARAMETERS);
        thrust::device_vector<double> popScores(POPULATION_SIZE);
        thrust::device_vector<double> newPopulation(POPULATION_SIZE * N_PARAMETERS);

        // Create raw pointers to the arrays createde by thrust.  When using 
        // device calls outside the thrust library, this is the appropriate input
        double* popPtr = thrust::raw_pointer_cast(&population[0]);
        double* newPopPtr = thrust::raw_pointer_cast(&newPopulation[0]);
        double* scoresPtr = thrust::raw_pointer_cast(&popScores[0]);

        // Allocated random state for the initial population
        curandState_t* states;
        gpuErrchk(cudaMalloc((void**) &states, POPULATION_SIZE*SIZE_PARENT_POOL*sizeof(curandState_t)));

        // Create random state for the random potential parents for each generation
        int *randParents;
        gpuErrchk(cudaMalloc((void**) &randParents, POPULATION_SIZE*SIZE_PARENT_POOL*sizeof(int)));

        // Allocate array for the actual parents used to breed in each generation
        int *parentsPool_d;
        gpuErrchk(cudaMalloc((void**) &parentsPool_d, POPULATION_SIZE*sizeof(int)));

        // Create random states and array for the parameters needed to breed children;
        // - the probability of placing the parent directly into the next gen
        // - the crossover point for the child
        // - probability of mutating the child
        // - index to mutate the child
        curandState_t* childStates;
        gpuErrchk(cudaMalloc((void**) &childStates, POPULATION_SIZE*4*sizeof(curandState_t)));
        int *randParams_d;
        gpuErrchk(cudaMalloc((void**)&randParams_d, POPULATION_SIZE*4*sizeof(int)));

        // Parse Argument variables
        if (argc >= 2) {
                POPULATION_SIZE = atoi(argv[1]);
        }
        if (argc >= 3) {
                N_PARAMETERS = atoi(argv[2]);
        }
        if (argc >= 4) {
                TOTALTHREADS = atoi(argv[3]);
        }
        if (argc >= 5) {
                BLOCKSIZE = atoi(argv[4]);
        }

        // Cude Device Properties to see if the BLOCKSIZE can be handled
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (BLOCKSIZE > prop.maxThreadsPerBlock) {
                std::cout << "Device only supports a BLOCKSIZE of " << prop.maxThreadsPerBlock <<" threads/block" << std::endl;
                std::cout << "Try again with a smaller BLOCKSIZE" << std::endl;
                return 1;
        }

        std::cout << "Running with " << TOTALTHREADS << " threads and a BLOCKSIZE of ";
        std::cout << BLOCKSIZE << std::endl;

        int numBlocks = TOTALTHREADS/BLOCKSIZE;

        // validate command line arguments and re-size if needed
        if (TOTALTHREADS % BLOCKSIZE != 0) {
                ++numBlocks;
                TOTALTHREADS = numBlocks*BLOCKSIZE;

                printf("Warning: Total thread count is not evenly divisible by the block size\n");
                printf("The total number of threads will be rounded up to %d\n", TOTALTHREADS);
        }
        std::cout << "Running " << numBlocks << " total blocks." << std::endl;

        // initialize random generator for the initial population
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, .1);

        // Generate initial random population representing generation 0
        thrust::counting_iterator<unsigned int> index_sequence_begin(0);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + POPULATION_SIZE * N_PARAMETERS,
                          population.begin(),
                          prg(MIN, MAX));

        // Evaluate every member of the population and find the most fit individual
        score<<<TOTALTHREADS, BLOCKSIZE>>>(POPULATION_SIZE, N_PARAMETERS, popPtr, scoresPtr);
        double best = *(thrust::max_element(popScores.begin(), popScores.end()));
        int best_index = min_index(popScores);

        std::cout << "Initial generation best score: " << best << " at index: " << best_index << "   ";
        for (int i=0; i<N_PARAMETERS; i++) {
                std::cout << population[best_index * N_PARAMETERS + i] << " ";
        }
        std::cout << std::endl;

        // Create successive generations until convergence is achieved.
        while (best < .999) {
                // Create a new set of random parents to breed into the next generation
                init<<<POPULATION_SIZE*SIZE_PARENT_POOL, 1>>>(time(0), states);
                gpuErrchk(cudaPeekAtLastError());
                setRandom<<<POPULATION_SIZE*SIZE_PARENT_POOL, 1>>>(states, randParents, POPULATION_SIZE);
                gpuErrchk(cudaPeekAtLastError());
                pickParents<<<TOTALTHREADS, BLOCKSIZE>>>(POPULATION_SIZE, N_PARAMETERS, randParents, scoresPtr, parentsPool_d);
                gpuErrchk(cudaPeekAtLastError());

                // Generate new random states for the breeding parameters
                init<<<POPULATION_SIZE*4, 1>>>(time(0), childStates);
                gpuErrchk(cudaPeekAtLastError());
                setRandom<<<POPULATION_SIZE*4, 1>>>(childStates, randParams_d, 100);
                gpuErrchk(cudaPeekAtLastError());

                // Generate new random parameters for breeding and child mutation
                thrust::device_vector<double> mutations(POPULATION_SIZE);
                thrust::counting_iterator<unsigned int> index_sequence_begin(0);
                thrust::transform(index_sequence_begin,
                                  index_sequence_begin + POPULATION_SIZE,
                                  mutations.begin(),
                                  normal(0.0, 0.001));
                double* mutPtr = thrust::raw_pointer_cast(&mutations[0]);

                // Breed members and copy over to the new generation
                breedGeneration<<<TOTALTHREADS, BLOCKSIZE>>>(POPULATION_SIZE, N_PARAMETERS, randParams_d, popPtr, newPopPtr, parentsPool_d, mutPtr);
                gpuErrchk(cudaPeekAtLastError());
                thrust::copy(thrust::device, newPopulation.begin(), newPopulation.end(), population.begin());

                // Evaluate all members and identify the most fit individual
                score<<<TOTALTHREADS, BLOCKSIZE>>>(POPULATION_SIZE, N_PARAMETERS, popPtr, scoresPtr);
                gpuErrchk(cudaPeekAtLastError());
                best = *(thrust::min_element(popScores.begin(), popScores.end()));
                best_index = min_index(popScores);

                std::cout << "Bred generation " << generation << " Best score: " << best << " at index: " << best_index << "   ";
                for (int i=0; i<N_PARAMETERS; i++) {
                        std::cout << population[best_index * N_PARAMETERS + i] << " ";
                }
                std::cout << std::endl;

                // Keep track of how many generations have occured. 
                generation++;
        }

        cudaFree(states);
        cudaFree(randParents);
        cudaFree(parentsPool_d);
        cudaFree(childStates);
        cudaFree(randParams_d);

        return 0;
}

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

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

const double MAX = 5.12;
const double MIN = -5.12;
const int n_population = 1000;
const int n_parameters = 100;


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


__host__ double evaluateMember(int index, double *array) {
        double sum = 0;
        for (int i=0; i<n_parameters; i++) {
                sum += pow(array[index*n_parameters+i] - 0.5, 2);
        }
        return sqrt(sum);
}

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
                                value += source[i*np+p];
                        }

                        value = (double) std::sqrt( (double) value);
                        score[i] = (double) 1.0 / (double) (value+1.0);
                }
        }
}

template <class T>
void evaluateGeneration(thrust::device_vector<T>& popScores, thrust::device_vector<T>& population) {
        thrust::device_vector<double> temp(population.size());
        thrust::copy(thrust::device, population.begin(), population.end(), temp.begin());

        thrust::transform(temp.begin(), temp.end(), temp.begin(), minus5<double>());
        thrust::transform(temp.begin(), temp.end(), temp.begin(), square<double>());

        double* tempPtr = thrust::raw_pointer_cast(&temp[0]);
        double* scoresPtr = thrust::raw_pointer_cast(&popScores[0]);

        score<<<512, 256>>>(n_population, n_parameters, tempPtr, scoresPtr);

        // for (int i=0; i<n_population; i++) {
        //         // std::cout << "Individual " << i << ":  ";
        //         // for (int offset=0; offset<n_parameters; offset++) {
        //         //         std::cout << population[i*n_parameters+offset] << " ";
        //         // }
        //
        //         double score = std::sqrt(thrust::reduce(temp.begin()+i*n_parameters, temp.begin()+i*n_parameters+n_parameters, (double) 0, thrust::plus<double>()));
        //         popScores[i] = (double) 1.0 / (double) (score+1.0);
        //
        //         // std::cout << " - Score: " << popScores[i] << endl;
        // }
}

__host__ thrust::device_vector<double> breed(thrust::device_vector<double> parentA, thrust::device_vector<double> parentB, int crossover){
        thrust::device_vector<double> child(n_parameters);

        thrust::copy(thrust::device, parentA.begin(), parentA.begin()+crossover, child.begin());
        thrust::copy(thrust::device, parentB.begin()+crossover, parentB.end(), child.begin()+crossover);

        return child;
}


__host__ void printMember(thrust::device_vector<double> member){
        cout << "Member: ";
        for (int i=0; i<n_parameters; i++) {
                cout << member[i] << " ";
        }
        cout << endl;
}


// We need this function to define how to sort
// the vector. We will pass this function into the
// third parameter and it will tell it to sort descendingly.
bool reverseSort(double i, double j) {
        return i > j;
}

int main(int argc, char** argv) {
        /* initialize random seed for timing purposes */
        //srand (time(NULL));
        srand (static_cast <unsigned> (time(0)));
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, .01);

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
        evaluateGeneration(popScores, population);
        double best = *(thrust::max_element(popScores.begin(), popScores.end()));
        int best_index = min_index(popScores);

        std::cout << "Initial generation " << generation << " Best score: " << best << " at index: " << best_index << "   ";
        for (int i=0; i<n_parameters; i++) {
                std::cout << population[best_index * n_parameters + i] << " ";
        }
        std::cout << endl;

        // begin timing functions
        float elapsedTime;
        cudaEvent_t startEvent, stopEvent;
        // Create start and stop events
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEvent_t startEvent2, stopEvent2;
        // Create start and stop events
        cudaEventCreate(&startEvent2);
        cudaEventCreate(&stopEvent2);

        while (best < .99) {
                // Time array creation
                cudaEventRecord(startEvent, 0);

                thrust::device_vector<double> newPopulation(n_population * n_parameters);

                // Time array creation
                cudaEventRecord(startEvent2, 0);

                int parentsPool[n_population];
                for (int n=0; n<n_population; n+=2) {
                        int pool[7];
                        double scores[7];

                        for (int i=0; i<7; i++) {
                                pool[i] = (rand()%n_population);
                                scores[i] = popScores[pool[i]];
                        }

                        // for (int y=0; y<7; y++) {
                        //         std::cout << " " << scores[y];
                        // }
                        // std::cout << endl;

                        std::sort(scores, scores+7, reverseSort);

                        // for (int y=0; y<7; y++) {
                        //   std::cout << " " << scores[y];
                        // }
                        // std::cout << endl;
                        // std::cout << "------------" << endl;

                        double parent_a_score = scores[0];
                        double parent_b_score = scores[1];

                        for (int s=0; s<7; s++) {
                                if (popScores[pool[s]] == parent_a_score) {
                                        parentsPool[n] = pool[s];
                                }
                                if (popScores[pool[s]] == parent_b_score) {
                                        parentsPool[n+1] = pool[s];
                                }
                        }
                }

                // Create stop events
                cudaEventRecord(stopEvent2, 0);
                cudaEventSynchronize(stopEvent2);

                // Print total elapsted seconds
                cudaEventElapsedTime(&elapsedTime, startEvent2, stopEvent2);
                //std::cout << "Picking parents took " << elapsedTime/1000 << " (seconds)" << endl;

                cudaEventRecord(startEvent2, 0);

                // parent arrays
                thrust::device_vector<double> parentA(n_parameters);
                thrust::device_vector<double> parentB(n_parameters);
                // children
                thrust::device_vector<double> childA(n_parameters);
                thrust::device_vector<double> childB(n_parameters);
                for (int n=0; n<n_population; n+=2) {
                        //cout << "Parent A: " << parentsPool[n] << " Score: " << popScores[parentsPool[n]];
                        //cout <<" Parent B: " << parentsPool[n+1] << " Score: " << popScores[parentsPool[n]];
                        //cout << endl;

                        int indexA = parentsPool[n];
                        int indexB = parentsPool[n+1];
                        int random = rand()%10;
                        if (random == 1) {
                                thrust::copy(thrust::device, population.begin()+indexA*n_parameters, population.begin()+indexA*n_parameters+n_parameters, newPopulation.begin()+n*n_parameters);
                                thrust::copy(thrust::device, population.begin()+indexB*n_parameters, population.begin()+indexB*n_parameters+n_parameters, newPopulation.begin()+n*n_parameters+n_parameters);

                        } else {
                                int crossover = rand()%n_parameters;

                                //thrust::device_vector<double> parentA(n_parameters);
                                thrust::copy(thrust::device, population.begin()+indexA*n_parameters, population.begin()+indexA*n_parameters+n_parameters, parentA.begin());

                                //thrust::device_vector<double> parentB(n_parameters);
                                thrust::copy(thrust::device, population.begin()+indexB*n_parameters, population.begin()+indexB*n_parameters+n_parameters, parentB.begin());

                                childA = breed(parentA, parentB, crossover);
                                childB = breed(parentB, parentA, crossover);

                                if (rand()%100 < 5) {
                                        int randIndex = rand()%n_parameters;
                                        double newval = childA[randIndex] += distribution(generator);
                                        childA[randIndex] = std::min(newval, MAX);
                                        childA[randIndex] = std::max(newval, MIN);
                                }

                                if (rand()%100 < 5) {
                                        int randIndex = rand()%n_parameters;
                                        double newval = childB[randIndex] += distribution(generator);
                                        childB[randIndex] = std::min(newval, MAX);
                                        childB[randIndex] = std::max(newval, MIN);
                                }



                                //for (int m=0; m<n_parameters; m++) {
                                //  cout << childA[m] << " ";
                                //}
                                //cout << endl;

                                //printMember(childA);
                                //printMember(childB);

                                thrust::copy(thrust::device, childA.begin(), childA.end(), newPopulation.begin()+n*n_parameters);
                                thrust::copy(thrust::device, childB.begin(), childB.end(), newPopulation.begin()+n*n_parameters+n_parameters);
                        }


                }
                // Create stop events
                cudaEventRecord(stopEvent2, 0);
                cudaEventSynchronize(stopEvent2);

                // Print total elapsted seconds
                cudaEventElapsedTime(&elapsedTime, startEvent2, stopEvent2);
                //std::cout << "Creating children took " << elapsedTime/1000 << " (seconds)" << endl;


                // Time array creation
                cudaEventRecord(startEvent2, 0);

                thrust::copy(thrust::device, newPopulation.begin(), newPopulation.end(), population.begin());

                // Create stop events
                cudaEventRecord(stopEvent2, 0);
                cudaEventSynchronize(stopEvent2);

                // Print total elapsted seconds
                cudaEventElapsedTime(&elapsedTime, startEvent2, stopEvent2);
                //std::cout << "First copy " << elapsedTime/1000 << " (seconds)" << endl;

                cudaEventRecord(startEvent2, 0);
                evaluateGeneration(popScores, population);
                // Print total elapsted seconds
                // Create stop events
                cudaEventRecord(stopEvent2, 0);
                cudaEventSynchronize(stopEvent2);
                cudaEventElapsedTime(&elapsedTime, startEvent2, stopEvent2);
                //std::cout << "compute scores " << elapsedTime/1000 << " (seconds)" << endl;

                best = *(thrust::min_element(popScores.begin(), popScores.end()));
                best_index = min_index(popScores);

                // Create stop events
                cudaEventRecord(stopEvent2, 0);
                cudaEventSynchronize(stopEvent2);

                // Print total elapsted seconds
                cudaEventElapsedTime(&elapsedTime, startEvent2, stopEvent2);
                //std::cout << "Copy to new gen took " << elapsedTime/1000 << " (seconds)" << endl;

                std::cout << "Bred generation " << generation << " Best score: " << best << " at index: " << best_index << "   ";
                //for (int i=0; i<n_parameters; i++) {
                //        std::cout << population[best_index * n_parameters + i] << " ";
                //}
                std::cout << endl;

                // Create stop events
                cudaEventRecord(stopEvent, 0);
                cudaEventSynchronize(stopEvent);

                // Print total elapsted seconds
                cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
                //std::cout << "Generation took " << elapsedTime/1000 << " (seconds)" << endl;

                generation++;
        }


        return 0;
}

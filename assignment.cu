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
                //thrust::default_random_engine rng;
                thrust::default_random_engine rng( 8675309 );
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
                sum += pow(array[index*n_parameters+i] - 5, 2);
        }
        return sqrt(sum);
}

// return the biggest of two tuples
template <class T>
struct smaller_tuple {
        __device__ __host__
        thrust::tuple<T,int> operator()(const thrust::tuple<T,int> &a, const thrust::tuple<T,int> &b)
        {
                if (a < b) return a;
                else return b;
        }

};

template <class T>
int min_index(thrust::device_vector<T>& vec) {

        // create implicit index sequence [0, 1, 2, ... )
        thrust::counting_iterator<int> begin(0); thrust::counting_iterator<int> end(vec.size());
        thrust::tuple<T,int> init(vec[0],0);
        thrust::tuple<T,int> smallest;

        smallest = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(vec.begin(), begin)), thrust::make_zip_iterator(thrust::make_tuple(vec.end(), end)),
                                  init, smaller_tuple<T>());
        return get<1>(smallest);
}


__host__ thrust::device_vector<float> pickParentSet(thrust::device_vector<float> population, thrust::device_vector<float> scores){
        thrust::device_vector<float> parents(n_population/2);
        thrust::fill(parents.begin(), parents.end(), 0.0);

        //int index = min_index(scores);

        return parents;
}

__host__ thrust::device_vector<float> evaluateGeneration(thrust::device_vector<float> population){
        thrust::device_vector<float> popScores(n_population);

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

__host__ thrust::device_vector<float> breed(thrust::device_vector<float> parentA, thrust::device_vector<float> parentB, int crossover){
        thrust::device_vector<float> child(n_parameters);
        //thrust::fill(child.begin(), child.end(), 1.0);

        thrust::copy(thrust::device, parentA.begin(), parentA.begin()+crossover, child.begin());
        thrust::copy(thrust::device, parentB.begin()+crossover, parentB.end(), child.begin()+crossover);

        return child;
}


__host__ void printMember(thrust::device_vector<float> member){
        cout << "Member: ";
        for (int i=0; i<n_parameters; i++) {
          cout << member[i] << " ";
        }
        cout << endl;
}


// We need this function to define how to sort
// the vector. We will pass this function into the
// third parameter and it will tell it to sort descendingly.
bool reverseSort(float i, float j) {
        return i > j;
}

int main(int argc, char** argv) {
        /* initialize random seed for timing purposes */
        //srand (time(NULL));
        srand (static_cast <unsigned> (time(0)));
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0, .01);
        
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
        while (generation < 200) {
                thrust::device_vector<float> newPopulation(n_population * n_parameters);

                thrust::device_vector<float> popScores = evaluateGeneration(population);

                float best = *(thrust::min_element(popScores.begin(), popScores.end()));
                int best_index = min_index(popScores);

                std::cout << "Bred generation " << generation << " Best score: " << best << " at index: " << best_index << "   ";
                for (int i=0; i<n_parameters; i++){
                  std::cout << population[best_index * n_parameters + i] << " ";
                } 
                std::cout << endl;


                int parentsPool[n_population];
                for (int n=0; n<n_population; n+=2) {
                        int pool[7];
                        float scores[7];

                        for (int i=0; i<7; i++) {
                                pool[i] = (rand()%n_population);
                                scores[i] = popScores[pool[i]];
                        }
                        
                        // for (int y=0; y<7; y++) {
                        //   std::cout << " " << scores[y];
                        // }
                        // std::cout << endl;
                        
                        std::sort(scores, scores+7);
                        // 
                        // for (int y=0; y<7; y++) {
                        //   std::cout << " " << scores[y];
                        // }
                        // std::cout << endl;
                        // std::cout << "------------" << endl;

                        float parent_a_score = scores[0];
                        float parent_b_score = scores[1];

                        for (int s=0; s<7; s++) {
                                if (popScores[pool[s]] == parent_a_score) {
                                        parentsPool[n] = pool[s];
                                }
                                if (popScores[pool[s]] == parent_b_score) {
                                        parentsPool[n+1] = pool[s];
                                }
                        }
                }

                for (int n=0; n<n_population; n+=2) {
                        //cout << "Parent A: " << parentsPool[n] << " Score: " << popScores[parentsPool[n]];
                        //cout <<" Parent B: " << parentsPool[n+1] << " Score: " << popScores[parentsPool[n]];
                        //cout << endl;

                        int indexA = parentsPool[n];
                        int indexB = parentsPool[n+1];
                        int random = rand()%10;
                        if (random < 1) {
                          for (int k=0; k<n_parameters; k++) {
                            newPopulation[n*n_parameters+k] = population[indexA*n_parameters+k];
                            newPopulation[n+1*n_parameters+k] = population[indexB*n_parameters+k];
                          }
                        } else {
                          int crossover = rand()%n_parameters;
                          
                          thrust::device_vector<float> parentA(n_parameters);
                          thrust::copy(thrust::device, population.begin()+indexA*n_parameters, population.begin()+indexA*n_parameters+n_parameters, parentA.begin());
                          
                          thrust::device_vector<float> parentB(n_parameters);
                          thrust::copy(thrust::device, population.begin()+indexB*n_parameters, population.begin()+indexB*n_parameters+n_parameters, parentB.begin());
                          
                          thrust::device_vector<float> childA = breed(parentA, parentB, crossover);
                          thrust::device_vector<float> childB = breed(parentA, parentB, crossover);
                          
                          if (rand()%100 < 5) {
                            int randIndex = rand()%n_parameters;
                            double newval = childA[randIndex] += distribution(generator);
                            childA[randIndex] = std::min(newval, 5.12);
                            childA[randIndex] = std::max(newval, -5.12);
                          }

                          if (rand()%100 < 5) {
                            int randIndex = rand()%n_parameters;
                            double newval = childB[randIndex] += distribution(generator);
                            childB[randIndex] = std::min(newval, 5.12);
                            childB[randIndex] = std::max(newval, -5.12);
                          }
                          //for (int m=0; m<n_parameters; m++) {
                          //  cout << childA[m] << " ";
                          //}
                          //cout << endl;
                          
                          //printMember(childA);
                          //printMember(childB);
                          
                          thrust::copy(thrust::device, childA.begin(), childA.end(), newPopulation.begin()+n*n_parameters);
                          thrust::copy(thrust::device, childB.begin(), childB.end(), newPopulation.begin()+n+1*n_parameters);
                        }


                }



                thrust::copy(thrust::device, newPopulation.begin(), newPopulation.end(), population.begin());

                generation++;
        }


        return 0;
}

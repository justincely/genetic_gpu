# Genetic Algorithm for Function Optimization

This project uses a genetic approach to minimize a specific function with N parameters.  The general 
approach is:

- Generate an initial population of randomly created members
- Evaluate the fitness of all the members against the function to minimize
- "Breed" the parents to create a child generation
    - 10% of the time, a parent just moves into the next generation without producing children
    - 90% of the time:
        - Children take a random half of their parameters from each parent
        - Children are randomly mutated 5% of the time
- Iterate and continue until the function is minimized to a given tolerance (.999)
 

## Build and Run
- `make`
- `./assignment.exe <n_population> <n_parameters> <n_threads> <block_size>`
  - *note* the `n_threads` and `block_size` does not pass to _all_ GPU functions.  Many operations are preformed by the Thrust library, which controls it's threading internally.

## Example Timings:
### Time to 2000 Generations
1. 1000 10: 0m9.878s
2. 2000 10: 0m19.661s 
3. 5000 10: 1m28.506s
4. 10000 10: 4m46.643s
5. 1000 20: 0m10.339s
6. 1000 30: 0m10.734s
7. 1000 50: 0m11.556s
8. 1000 100: 0m13.522s
9. 1000 1000: 0m50.640s

## Time to Convergence:
1. 1000 10: 0m4.364s
2. 2000 10: 0m4.522s
3. 5000 10: 0m10.353s
4. 10000 10: 0m8.888s
5. 1000 20: 0m30.116s
6. 1000 30: 0m50.463s
9. 1000 50: 1m59.999s 
8. 1000 100: 8m41.778s

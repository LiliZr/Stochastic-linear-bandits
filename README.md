# Stochastic-linear-bandits

Implementation of Stochastic Linear Bandits using optimizations.

## How to run?
1) Try an example:
    - run ```python main.py``` to run algorithms mentionned in the file on the given dataset as an example.
    > Best parameters combination of each dataset and algorithm are given in "dict_params.json". 

2) Find the best parameters values:
    - First, run ```python params_dataset.py``` to run a specified algorithm on a certain dataset to try different values of parameters.

    - Then, run ```python summerize_results.py``` to find the best parameters and plot them

## Implementation details
### Algorithms
- Different algorithms were implemented:
1) Baseline algorithms
    - Random: Select an arm randomly at each iteration.
    - Optimal: Knows the exact model parameter (it selects the best arm possible). 
    - ConfidenceBall1: Uses an L1 ellipsoid around the estimator.
    - LinUCB: Compute ucb directly as mentionned in section 19.3.1 in [1]
2) Existing optimized algorihtms:
    - CBRAP: Random Projections [2]
    - SOFUL: Frequent directions [3]
    - CBSCFD: SCFD (based on frequent directions) with better regularization [4]
3) Our approach:
    - ConfidenceBall1_FJLT: Use Fast Johnson-Lindenstrauss Transform (FJLT) to reduce data dimension.
- Each algorithm handles a finite action set case.

### Datasets
5 real world datasets have been processed in order to match the bandit problem:
- [Amazon books](https://amazon-reviews-2023.github.io/data_processing/index.html)
- [Mnist digits](https://pjreddie.com/projects/mnist-in-csv/)
- [Movielens](https://grouplens.org/datasets/movielens/)
- [Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- [Yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49)
- Random synthetic dataset: the model parameter and arms are sampled from a normal distribution.

 
## References

[1] ["Bandit Algorithms Book"](https://tor-lattimore.com/downloads/book/book.pdf)

[2] ["CBRAP: Contextual Bandits with RAndom Projection"](https://ojs.aaai.org/index.php/AAAI/article/view/10888)

[3] ["Efficient Linear Bandits through Matrix Sketching"](https://proceedings.mlr.press/v89/kuzborskij19a.html)

[4] ["Efficient and Robust High-Dimensional Linear Contextual Bandits"](https://www.ijcai.org/proceedings/2020/588)

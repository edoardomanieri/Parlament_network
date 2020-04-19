# PARLIAMENT NETWORK

Thesis project that aims to uncover latent networks among italian politicians.

# Project description

- Gather data of sponsorships among politicians of the italian parliament [Briatte, 2016].
- Obtain the adjacency matrices for each legislation.
- Use the Infinite Relational Model to model the parliament network.
- Implement Gibbs Sampling to obtain the distribution of the clusters inside the parliament.

# Project structure

- _irm.py_ : file containing different models used to implement Gibbs Sampling for the Infinite Relational Model.
- _network.py_: main execution of the model
- _preprocessing.py_: preprocessing of the parliament csv data
- _import_data_: folder containing the modified R scripts used to gather data
- _gibbs_sampling_matlab.m_: paper implementation of Gibbs Sampling in Matlab

# CRP-Beta-Bernoulli Model

Parameters:

- Beta: alpha = 1, beta = 1 => uniform distribution in (0,1)
- CRP: parameter theta chosen such that the expected value is equal to the number of parties. (Used newton's method)

# Point estimate

Through gibbs sampling is it possible to draw from the posterior distribution of the partitions. In this case obtaining the point estimate is not straightforward as in other cases since we do not have a "summary" measure for partitions. The procedure that we followed to obtain such point estimate is:

- We obtained a co-clustering matrix M such that for each point M(x,y) we have the relative frequency of the number of times they were in the same partition from the posterior draws.
- We used agglomerative hierarchical clustering with the co-clustering matrix as the distance matrix, and the posterior average number of clusters as number of clusters.

# Walking through gibbs sampling Python code

### Matrices involved in counting number of links between components

1. `z[nn, :].T@X_nn_nn` : matrix components-nodes with number of links
2. `z[nn, :].T@X_nn_nn@z[nn, :]` : matrix components-components with number of links (double count of some links ex: [A->B B->A])

### Matrices involved in counting number of no-links between components

1. `m@m.T` : matrix components-components with total number of possible links among components
2. `np.diag((m*(m+1)).ravel() / 2)` : matrix to remove from the count links of nodes with themselves (ex. [A->A])

# References

> Briatte, François. 2016. “Network Patterns of Legislative Collaboration in Twenty Parliaments.” Network Science, 4(2): 266–71. doi:10.1017/nws.2015.31

> Schmidt, Mikkel N. and Mørup, Morten. 2013. "Non-parametric Bayesian modeling of complex networks"

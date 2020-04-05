# PARLIAMENT NETWORK

Thesis project that aims to uncover latent alliances among italian politicians.

# Structure of the project

1. Gather data of sponsorships among politicians of the italian parliament by modifying R scripts [Briatte, 2016]. 
2. Obtain the adjacency matrices for each legislation and build a network model.
3. Use the Infinite Relational Model to model the parliament network.
4. Implement Gibbs Sampling to obtain the distribution of the clusters inside the parliament.
5. ...

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
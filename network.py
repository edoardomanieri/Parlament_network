import click
import pandas as pd
from utils import co_clustering_matrix, build_output, point_estimate
import time
import DP
import PY


def DP_model(X, n_politicians, n_parties, data_desc, a_beta, b_beta,
             theta, gibbs_sweeps, burn_in):
    start_time = time.time()
    if theta == 0:
        # Get theta using newton's method
        theta = DP.crp_parameters(n_politicians, n_parties, 10000)
    print("Got CRP parameters\n")
    model = "DP-Beta-Bernoulli"
    params = {
        'a': a_beta,
        'b': b_beta,
        'theta': theta,
        'T': gibbs_sweeps
    }

    # Compute Z matrices
    Z = DP.beta_bernoulli_irm(X=X, **params)
    print("computed Z matrices\n")

    # Compute co-clustering matrix
    co_clust_matrix, avg_n_clusters = co_clustering_matrix(Z, burn_in_factor=burn_in)
    print("computed co-clutering matrix\n")
    print(f"average number of clusters {avg_n_clusters}\n")

    # Get point estimate
    labels = point_estimate(co_clust_matrix, avg_n_clusters)

    final_time = time.time()
    diff_time = final_time - start_time
    build_output(data_desc, model, params, co_clust_matrix,
                 avg_n_clusters, labels, diff_time)


def PY_model(X, n_politicians, n_parties, data_desc, a_beta, b_beta, theta, alpha, gibbs_sweeps, burn_in):
    start_time = time.time()
    try:
        if theta == 0:
            theta, alpha = PY.crp_parameters(n_politicians, n_parties, 10000)
    except ValueError as e:
        return
    print("Got CRP parameters\n")
    model = "PY-Beta-Bernoulli"
    params = {
        'a': a_beta,
        'b': b_beta,
        'theta': theta,
        'alpha': alpha,
        'T': gibbs_sweeps
    }
    Z = PY.beta_bernoulli_irm(X=X, **params)
    print("computed Z matrices\n")
    co_clust_matrix, avg_n_clusters = co_clustering_matrix(Z, burn_in_factor=burn_in)
    print("computed co-clutering matrix\n")
    print(f"average number of clusters {avg_n_clusters}\n")
    labels = point_estimate(co_clust_matrix, avg_n_clusters)
    final_time = time.time()
    diff_time = final_time - start_time
    build_output(data_desc, model, params, co_clust_matrix,
                 avg_n_clusters, labels, diff_time)


@click.command()
@click.option('--process',
              default='DP',
              help='DP o PY process', type=str)
@click.option('--a_beta',
              default=1,
              help='alpha parameter of beta distributions', type=float)
@click.option('--b_beta',
              default=1,
              help='beta parameter of beta distributions', type=float)
@click.option('--theta',
              default=0,
              help='CRP theta parameter', type=float)
@click.option('--alpha',
              default=0,
              help='CRP alpha parameter', type=float)
@click.option('--gibbs_sweeps',
              default=5,
              help='number of Gibbs sweeps', type=int)
@click.option('--leg',
              default=9,
              help='legislature number', type=int)
@click.option('--burn_in',
              default=1,
              help='burn in factor', type=int)
def main(process, a_beta, b_beta, theta, alpha, gibbs_sweeps, leg, burn_in):
    if leg > 17 or leg < 9:
        print("Leg number not valid\n")
        return
    data_desc = ['Italian parliament', 'camera', f'{leg}leg']

    # Reading adjacency matrix
    adj = pd.read_csv(f"../data/output/adjmat_leg{leg}.csv", index_col=0)
    X = adj.values

    # Reading parties info
    df_parties = pd.read_csv(f"../data/output/parties_number.csv", index_col=0)

    # Get number of parties
    n_parties = df_parties.loc[df_parties['legislature'] == leg, 'number_of_parties'].values[0]
    print(f"number of parties: {n_parties}\n")

    # Get number of politicians
    n_politicians = len(adj)
    print(f"number of politicians: {n_politicians}\n")

    print("run model...\n")
    if process == 'DP':
        DP_model(X, n_politicians, n_parties, data_desc, a_beta, b_beta,
                 theta, gibbs_sweeps, burn_in)
    if process == 'PY':
        PY_model(X, n_politicians, n_parties, data_desc, a_beta, b_beta,
                 theta, alpha, gibbs_sweeps, burn_in)


if __name__ == "__main__":
    main()

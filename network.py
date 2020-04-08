import click
import pandas as pd
from irm import beta_bernoulli, crp_parameter
from utils import co_clustering_matrix, build_output, point_estimate
import time


@click.command()
@click.option('--a_beta',
              default=1,
              help='alpha parameter of beta distributions', type=float)
@click.option('--b_beta',
              default=1,
              help='beta parameter of beta distributions', type=float)
@click.option('--theta',
              default=0,
              help='CRP parameter', type=float)
@click.option('--gibbs_sweeps',
              default=5,
              help='number of Gibbs sweeps', type=int)
@click.option('--leg',
              default=9,
              help='legislature number', type=int)
def main(a_beta, b_beta, theta, gibbs_sweeps, leg):
    start_time = time.time()
    if leg > 17 or leg < 9:
        print("Leg number not valid\n")
        return
    adj = pd.read_csv(f"./data/output/adjmat_leg{leg}.csv").drop(columns=['Unnamed: 0'])
    data_desc = ['Italian parliament', 'camera', f'{leg}leg']
    X = adj.values
    df_parties = pd.read_csv(f"./data/output/parties_number.csv")
    print("read csv files\n")
    if theta == 0:
        n_parties = df_parties.loc[df_parties['legislature'] == leg, 'number_of_parties'].values[0]
        n_politicians = len(adj)
        theta = crp_parameter(n_politicians, n_parties, 10000)
    print("Got CRP parameter\n")
    model = "CRP-Beta-Bernoulli"
    params = {
        'a': a_beta,
        'b': b_beta,
        'theta': theta,
        'T': gibbs_sweeps
    }
    Z = beta_bernoulli(X=X, **params)
    print("computed Z matrices\n")
    co_clust_matrix, avg_n_clusters = co_clustering_matrix(Z, burn_in_factor=1)
    print("computed co-clutering matrix\n")
    print(f"average number of clusters {avg_n_clusters}\n")
    labels = point_estimate(co_clust_matrix, avg_n_clusters)
    final_time = time.time()
    diff_time = final_time - start_time
    build_output(data_desc, model, params, co_clust_matrix,
                 avg_n_clusters, labels, diff_time)


if __name__ == "__main__":
    main()

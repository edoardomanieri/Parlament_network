import click
import numpy as np
import pandas as pd
from irm import beta_bernoulli, crp_parameter
from itertools import islice
from pathlib import Path
from datetime import datetime
from operator import itemgetter
from collections import defaultdict
from itertools import combinations


def get_partitions(Z, burn_in_factor=2):
    """
    Get partitions dictionary from list of partitioning matrices sampled from
    posterior

    Parameters
    ----------
    Z : list of np.arrays
        list of partitioning matrices sampled from posterior
    burn_in_factor : int, optional
        burn in factor for gibbs sampling, by default 2

    Returns
    -------
    dict
        dict with partitions as keys and number of times it occurs as values
    """

    burn_in = int(len(Z) - (len(Z)/burn_in_factor))
    Z = Z[burn_in:]
    res = defaultdict(int)
    for z in Z:
        df = pd.DataFrame(data=np.argwhere(z), columns=['node', 'group'])
        df = df.groupby('group')['node'].apply(tuple)
        tmp = tuple(sorted(df.values.tolist(), key=itemgetter(0)))
        res[tmp] += 1
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1],
                                   reverse=True)}
    return res


def co_clustering_matrix(Z, burn_in_factor=2):
    """
    Get co-clustering matrix based on relative frequencies on gibbs sampling

    Parameters
    ----------
    Z : list of np.arrays
        list of partitioning matrices sampled from posterior
    burn_in_factor : int, optional
        burn in factor for gibbs sampling, by default 2

    Returns
    -------
    np.array
        co-clustering matrix
    """

    burn_in = int(len(Z) - (len(Z)/burn_in_factor))
    Z = Z[burn_in:]
    res = np.zeros((Z[0].shape[0], Z[0].shape[0]))
    for z in Z:
        for col in z.T:
            indices = np.nonzero(col)[0]
            if len(indices) > 1:
                np_idx = np.array(list(combinations(indices, 2))).T
                np_idx1 = np_idx[0, :]
                np_idx2 = np_idx[1, :]
                res[np_idx1, np_idx2] += 1
    res /= len(Z)
    res += res.T
    np.fill_diagonal(res, 1)
    return res


def build_output_old(data_desc, model, params, partitions_dict):
    dt = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
    directory = f"./output/{data_desc[1]}/{data_desc[2]}/{dt}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    report_file = open(f"{directory}/report", "w+")
    report_file.write(f"data description: {data_desc[0]} - ")
    report_file.write(f"{data_desc[1]} - {data_desc[2]}\n")
    report_file.write(f"model: {model}\n")
    report_file.write(f"parameters:\n")
    for key, value in params.items():
        report_file.write(f"\t {key}: {value}\n")
    report_file.write(f"partitions:\n")
    report_file.write(f"(data presented as follows: number of cluster in ")
    report_file.write(f"particular configuration - number of times particular")
    report_file.write(f" configuration occured during sampling for the first ")
    report_file.write(f"five most commmon configurations\n")
    for key, value in partitions_dict.items():
        report_file.write(f"\t {len(key)}: {value}\n")
    report_file.close()
    directory += '/partitions'
    Path(directory).mkdir(parents=True, exist_ok=True)
    n = 5 if len(partitions_dict) > 5 else len(partitions_dict)
    first_five = dict(islice(partitions_dict.items(), n))
    for idx, key_value in enumerate(first_five.items()):
        df = pd.DataFrame(data=key_value[0])
        df.to_csv(f"{directory}/{idx}_part.csv")


def build_output(data_desc, model, params, co_clustering_matrix):
    dt = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
    directory = f"./output/{data_desc[1]}/{data_desc[2]}/{dt}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    report_file = open(f"{directory}/report", "w+")
    report_file.write(f"data description: {data_desc[0]} - ")
    report_file.write(f"{data_desc[1]} - {data_desc[2]}\n")
    report_file.write(f"model: {model}\n")
    report_file.write(f"parameters:\n")
    for key, value in params.items():
        report_file.write(f"\t {key}: {value}\n")
    report_file.close()
    df = pd.DataFrame(data=co_clustering_matrix)
    df.to_csv(f"{directory}/co_clust.csv")


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
    partitions = co_clustering_matrix(Z, burn_in_factor=1)
    print("computed co-clutering matrix\n")
    build_output(data_desc, model, params, partitions)


if __name__ == "__main__":
    main()

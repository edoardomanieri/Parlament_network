import numpy as np
import pandas as pd
from irm import beta_bernoulli
from itertools import islice
from pathlib import Path
from datetime import datetime
from operator import itemgetter
from collections import defaultdict


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


def build_output(data_desc, model, params, partitions_dict):
    dt = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
    directory = f"./output/{data_desc[1]}/{data_desc[2]}/{dt}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    report_file = open(directory + "/report", "w+")
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
    first_five = dict(islice(partitions.items(), n))
    for idx, key_value in enumerate(first_five.items()):
        df = pd.DataFrame(data=key_value[0])
        df.to_csv(f"{directory}/{idx}_part.csv")


if __name__ == "__main__":
    adj = pd.read_csv("./data/output/adjmat_leg9.csv").drop(columns=['Unnamed: 0'])
    data_desc = ['Italian parliament', 'camera', '9leg']
    X = adj.values
    model = "CRP-Beta-Bernoulli"
    params = {
        'a': 1,
        'b': 1,
        'A': 5,
        'T': 5
    }
    Z = beta_bernoulli(X=X, **params)
    partitions = get_partitions(Z, burn_in_factor=1)
    build_output(data_desc, model, params, partitions)

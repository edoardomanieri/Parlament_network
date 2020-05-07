import pandas as pd
from sknetwork.visualization import svg_graph
from IPython.display import SVG

adj = pd.read_csv(f"../data/output/adjmat_leg9.csv", index_col=0)
labels = pd.read_csv(f"output/camera/9leg/04-29-2020,17:46:50/final_labels.csv", index_col=0)
image = svg_graph(adj.values, labels=labels.values.T.squeeze())
SVG(image)
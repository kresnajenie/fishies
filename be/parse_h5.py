import scanpy as sc
import pandas as pd
import redis_test as re
from tqdm import tqdm
import json


def parse_h5(adata):
    # convert adata to df (normalized values)
    df = adata.to_df()
    # add the obs in the df
    df = pd.concat([df, adata.obs], axis=1)

    # add the brain projection coordinates in the df
    for x in range(adata.obsm["X_spatial"].shape[1]):
        df[f"X_spatial{x}"] = adata.obsm["X_spatial"][:,x]

    # add the umap coordinates in the df
    # iterate through all axis if there's more than 2, it accounts for it
    for x in range(adata.obsm["X_umap"].shape[1]):
        df[f"X_umap{x}"] = adata.obsm["X_umap"][:,x]

    # save to redis norm + obs + umap + spatial
    for col in tqdm(df):
        print(col)
        print()
        # re.save_to_redis(col, str(df[col].to_list()))
        re.save_to_redis(col, json.dumps(df[col].to_list()))
        # break

    # convert adata to df (raw values)
    df = pd.DataFrame(adata.obsm["X_raw"], columns=adata.to_df().columns, index=adata.obs.index)

    # save to redis raw
    for col in tqdm(df):
        re.save_to_redis(f"{col}_raw", json.dumps(df[col].to_list()))
        # break
    

    # also save the uns for coloring purposes
    re.save_to_redis("uns", str(adata.uns))

if __name__ == "__main__":
    adata = sc.read("../Data/4_week_full_labeled_celltype.h5ad")
    parse_h5(adata)

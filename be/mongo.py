from pymongo import MongoClient
import pandas as pd
import scanpy as sc

# Connect to MongoDB
client = MongoClient('mongodb://192.168.64.2:27017/')  # Change connection string if necessary

# Access the database
db = client['mydatabase']  # Change 'mydatabase' to the name of your database

# Access the collection (or create it if it doesn't exist)
collection = db['mycollection']  # Change 'mycollection' to the name of your collection

adata = sc.read("../Data/4_week_full_labeled_celltype.h5ad")

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

df_raw = pd.DataFrame(adata.obsm["X_raw"], columns=adata.to_df().columns, index=adata.obs.index)

for c in df.columns:
    # Define a document (key-value pair)
    document = {
        'name': c,
        'genes': df[c].tolist(),
        'raw': df_raw[c].tolist(),
        # Add more key-value pairs as needed
    }

    # Insert the document into the collection
    result = collection.insert_one(document)

    # Print the inserted document's ID
    print('Inserted document ID:', result.inserted_id)

# Close the MongoDB connection (optional, Python will close it automatically when the script ends)
client.close()


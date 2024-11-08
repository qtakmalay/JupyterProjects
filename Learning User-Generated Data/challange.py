import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split

# Check if TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("List of Physical Devices:", tf.config.list_physical_devices())

# Load data into DataFrame
train_inters = pd.read_csv(r'C:\Users\azatv\Jupyter\JupyterProjects\Learning User-Generated Data\lfm-challenge\lfm-challenge.inter_train', sep='\t')
test_inters = pd.read_csv(r'C:\Users\azatv\Jupyter\JupyterProjects\Learning User-Generated Data\lfm-challenge\lfm-challenge.inter_test', sep='\t')
items = pd.read_csv(r'C:\Users\azatv\Jupyter\JupyterProjects\Learning User-Generated Data\lfm-challenge\lfm-challenge.item', sep='\t')
users = pd.read_csv(r'C:\Users\azatv\Jupyter\JupyterProjects\Learning User-Generated Data\lfm-challenge\lfm-challenge.user', sep='\t')
test_indices = pd.read_csv(r'C:\Users\azatv\Jupyter\JupyterProjects\Learning User-Generated Data\lfm-challenge\test_indices.txt', sep='\t')


# Combine train and test interactions for full dataset
full_interactions = pd.concat([train_inters, test_inters])

# Check for null values
print(full_interactions.isnull().sum())
print(items.isnull().sum())
print(users.isnull().sum())

# Fill missing values in items and users
items.fillna('', inplace=True)
users.fillna('', inplace=True)

# Combine train and test interactions for full dataset
full_interactions = pd.concat([train_inters, test_inters])

# Convert user_id and item_id to strings
full_interactions['user_id'] = full_interactions['user_id'].astype(str)
full_interactions['item_id'] = full_interactions['item_id'].astype(str)

# Display the first few rows of each DataFrame to ensure data is loaded correctly
print(full_interactions.head())
print(items.head())
print(users.head())

# Convert DataFrame to TensorFlow Dataset
def df_to_tf_dataset(df):
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    return tf.data.Dataset.from_tensor_slices(dict(df))

train_df, test_df = train_test_split(full_interactions, test_size=0.2)
train_ds = df_to_tf_dataset(train_df).batch(4096)
test_ds = df_to_tf_dataset(test_df).batch(4096)

# Print to verify data is loaded correctly
print("Train Dataset Example:")
for example in train_ds.take(1):
    print(example)

print("Test Dataset Example:")
for example in test_ds.take(1):
    print(example)

class RecommenderModel(tfrs.Model):
    def __init__(self, user_ids, item_ids):
        super().__init__()
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
        self.item_lookup = tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None)
        self.user_embedding = tf.keras.layers.Embedding(len(user_ids) + 1, 32)
        self.item_embedding = tf.keras.layers.Embedding(len(item_ids) + 1, 32)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=train_ds.map(lambda x: (x["item_id"], self.item_embedding(self.item_lookup(x["item_id"])))).cache()
            )
        )

    def compute_loss(self, features, training=False):
        user_indices = self.user_lookup(features["user_id"])
        item_indices = self.item_lookup(features["item_id"])
        user_embeddings = self.user_embedding(user_indices)
        item_embeddings = self.item_embedding(item_indices)
        return self.task(user_embeddings, item_embeddings)

# Get unique user and item IDs
user_ids = full_interactions["user_id"].unique()
item_ids = full_interactions["item_id"].unique()

# Print to verify unique user and item IDs
print("Unique User IDs:", user_ids[:5])
print("Unique Item IDs:", item_ids[:5])

model = RecommenderModel(user_ids, item_ids)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Train the model
model.fit(train_ds, epochs=5)

# Evaluate the model
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    train_ds.map(lambda x: (x["item_id"], model.item_model(model.item_lookup(x["item_id"])))).cache()
)

# Generate recommendations for test users
user_ids = test_df["user_id"].unique()
recommendations = {user_id: index(tf.constant([user_id]))[1].numpy() for user_id in user_ids}

# Calculate nDCG
def get_ndcg(recommendations, test_interactions, k=10):
    from collections import defaultdict
    from math import log
    
    # Create a mapping of test interactions
    user_item_map = defaultdict(set)
    for _, row in test_interactions.iterrows():
        user_item_map[row["user_id"]].add(row["item_id"])
    
    ndcgs = []
    for user_id, recommended_items in recommendations.items():
        recommended_items = recommended_items[:k]
        dcg = sum([1 / log(i + 2, 2) if item in user_item_map[user_id] else 0 for i, item in enumerate(recommended_items)])
        idcg = sum([1 / log(i + 2, 2) for i in range(min(len(user_item_map[user_id]), k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs)

ndcg_score = get_ndcg(recommendations, test_df, k=10)
print(f'nDCG Score: {ndcg_score}')
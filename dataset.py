import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the fake reviewsdataset into a Pandas DataFrame
# Source: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
df = pd.read_csv("reviews_raw.csv")

# One hot encode the "label" column and create a new column "is_fake"
# Only one binary class is needed, so adding an is_real column is redundant
df["is_fake"] = (df["label"] == "CG").astype(int)

# Map the category column to numeric values for embedding
encoder = LabelEncoder()
df["category_id"] = encoder.fit_transform(df["category"])

# Save the preprocessed DataFrame to a CSV file
df.to_csv("reviews_ready.csv", index=False)
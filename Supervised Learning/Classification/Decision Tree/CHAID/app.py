import pandas as pd
from CHAID import Tree

# Sample dataset
data = pd.DataFrame({
    'Age': ['Young', 'Young', 'Middle', 'Senior', 'Senior', 'Senior', 'Middle', 'Young', 'Young', 'Senior', 'Young', 'Middle', 'Middle', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Credit_rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Excellent'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Map categorical values to numerical (required by CHAID)
for col in data.columns:
    data[col] = pd.factorize(data[col])[0]

# Features and target
features = ['Age', 'Income', 'Student', 'Credit_rating']
target = 'Buys_Computer'

# Build CHAID tree
tree = Tree.from_pandas_df(data, dict([(col, 'nominal') for col in features]), target, max_depth=3)

# Print the tree
tree.print_tree()

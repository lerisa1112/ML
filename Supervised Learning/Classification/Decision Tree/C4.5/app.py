import pandas as pd
import numpy as np
from math import log2

# Sample dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Entropy calculation
def entropy(data):
    labels = data['PlayTennis'].value_counts()
    ent = 0
    for count in labels:
        p = count / len(data)
        ent -= p * log2(p)
    return ent

# Gain Ratio Calculation
def gain_ratio(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    split_info = 0
    subset_entropy = 0

    for val in values:
        subset = data[data[attribute] == val]
        prob = len(subset) / len(data)
        subset_entropy += prob * entropy(subset)
        split_info -= prob * log2(prob)

    info_gain = total_entropy - subset_entropy
    return info_gain / split_info if split_info != 0 else 0

# Select best attribute using Gain Ratio
def best_attribute(data):
    attributes = data.columns[:-1]  # exclude target
    gain_ratios = {attr: gain_ratio(data, attr) for attr in attributes}
    return max(gain_ratios, key=gain_ratios.get)

# C4.5 Tree Construction
def build_tree(data, depth=0):
    labels = data['PlayTennis'].unique()
    if len(labels) == 1:
        return labels[0]

    if len(data.columns) == 1:
        return data['PlayTennis'].mode()[0]

    best_attr = best_attribute(data)
    tree = {best_attr: {}}

    for val in data[best_attr].unique():
        sub_data = data[data[best_attr] == val].drop(columns=[best_attr])
        tree[best_attr][val] = build_tree(sub_data, depth + 1)

    return tree

# Build and print tree
decision_tree = build_tree(df)
import pprint
pprint.pprint(decision_tree)

import pandas as pd
import math
import numpy as np

data = pd.read_csv("weather.csv")
features = [feat for feat in data]
features.remove("Class")

class Node:
    def __init__(self):
        self.child = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

def entropy(examples):
    y = 0.0
    n = 0.0
    
    for _, row in examples.iterrows():
        if row["Class"] == "Plays":
            y += 1
        elif row["Class"] == "Not_plays":
            n += 1
    if y == 0.0 or n == 0.0:
        return 0.0
    else:
        yes = y / (y + n)
        no = n /  (y + n)
        return -(yes * math.log(yes, 2) + no * math.log(no, 2))

def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)

    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain

def ID3(examples, attrs):
    root = Node()

    max_gain = 0
    max_feat = ""
    
    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    
    root.value = max_feat
    uniq = np.unique(examples[max_feat])
    # print(max_feat, uniq)
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["Class"])
            root.child.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            print("For",np.unique(subdata[max_feat])[0])
            child = ID3(subdata, new_attrs)
            dummyNode.child.append(child)
            root.child.append(dummyNode)
        printTree(root)
    return root

def printTree(root: Node, depth=0):
    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    for child in root.child:
        printTree(child, depth + 1)

root = ID3(data, features)
# printTree(root)
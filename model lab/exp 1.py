import math
import pandas as pd

data = {
    'Rating': ['High','Medium','Low','Medium'],
    'Delivery': ['Fast','Fast','Slow','Slow'],
    'Quality': ['Good','Average','Poor','Average'],
    'Sentiment': ['Positive','Positive','Negative','Negative']
}

df = pd.DataFrame(data)

def entropy(target):
    total = len(target)
    counts = target.value_counts()
    ent = 0
    for c in counts:
        p = c/total
        ent -= p * math.log2(p)
    return ent

print("Entropy of Dataset:", entropy(df['Sentiment']))

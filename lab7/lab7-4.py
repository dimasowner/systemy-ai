import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster

input_file = 'company_symbol_mapping.json'
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

start_date = "2003-07-03"
end_date = "2007-05-04"

print("Завантаження даних...")
data = yf.download(list(symbols), start=start_date, end=end_date)

quotes_diff = data['Close'] - data['Open']
X = quotes_diff.values.T 

mask = ~np.isnan(X).any(axis=1)
X = X[mask]
names = names[mask]

X /= X.std(axis=1)[:, np.newaxis]

edge_model = covariance.GraphicalLassoCV()
edge_model.fit(X.T)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

print("\nРезультати кластеризації:")
for i in range(num_labels + 1):
    print(f"Cluster {i+1} ==> {', '.join(names[labels == i])}")
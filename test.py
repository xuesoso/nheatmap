```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
import scripts, sklearn.decomposition, plot, sklearn.cluster
import matplotlib.pyplot as plt
import matplotlib as mpl
```

```python
df = scripts.simulate_data(nrows=120)
nrows, ncols = np.shape(df)
pc = sklearn.decomposition.PCA().fit(df)
dfr = pd.DataFrame(pc.transform(df)[:, 0], index=['sample '+str(x) for x in np.arange(1, nrows+1)],
        columns=['PC1'])
dfr['cell cluster'] = sklearn.cluster.KMeans(n_clusters=3).fit_predict(df).astype(str)
dfc = pd.DataFrame(pc.components_[0], index=['gene '+str(x) for x in
    np.arange(1, ncols+1)], columns=['PC score'])
dfc['gene cluster'] = sklearn.cluster.KMeans(n_clusters=3).fit_predict(df.T).astype(str)
cmaps={'cell cluster':'Paired', 'PC1':'RdYlGn', 'gene cluster':'inferno',
        'PC score':'gist_heat'}

g = plot.nheatmap(data=df, dfr=dfr, dfc=dfc, figsize=(8, 15),
        cmaps=cmaps, linewidths=0, showxticks=False)
g.hcluster()
# fig.savefig('./examples/example1.png', bbox_inches='tight')
```

```python
g = plot.nheatmap(data=df, dfr=dfr, dfc=dfc, figsize=(10, 15),
        cmaps=cmaps, linewidths=0, showxticks=False)
g.hcluster()
fig, plots = g.run()
# fig.savefig('./test.png', bbox_inches='tight')

t = g.legend_data['gene cluster']
t['mapper'](t['ulab'][0])


```

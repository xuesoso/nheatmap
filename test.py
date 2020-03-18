import pandas as pd
import numpy as np
import scripts, sklearn.decomposition, plot, sklearn.cluster
import matplotlib.pyplot as plt
import matplotlib as mpl

df = scripts.simulate_data(nrows=120)
nrows, ncols = np.shape(df)
pc = sklearn.decomposition.PCA().fit(df)
dfr = pd.DataFrame(pc.transform(df)[:, 0], index=['sample '+str(x) for x in np.arange(1, nrows+1)],
        columns=['PC1'])
dfr['cell cluster'] = sklearn.cluster.KMeans(n_clusters=20).fit_predict(df).astype(str)
dfc = pd.DataFrame(pc.components_[0], index=['gene '+str(x) for x in
    np.arange(1, ncols+1)], columns=['PC score'])
dfc['gene cluster'] = sklearn.cluster.KMeans(n_clusters=10).fit_predict(df.T).astype(str)
dfc['PC score 2'] = pc.components_[1]
cmaps={'cell cluster':'Paired', 'PC1':'RdYlGn', 'gene cluster':'inferno',
        'PC score':'gist_heat', 'PC score 2':'rainbow'}

g = plot.nheatmap(data=df, dfr=dfr, dfc=dfc, figsize=(10, 15),
        cmaps=cmaps, linewidths=0, showxticks=False)
g.hcluster()
fig, plots = g.run()
fig.savefig('./examples/example1.png', bbox_inches='tight')


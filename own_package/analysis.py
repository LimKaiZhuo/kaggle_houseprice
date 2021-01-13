import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



def analyze_randomsearchresults(results_dir_store, name_store):
    top_results = {}
    rmsle_results = {}
    for results_dir, name in zip(results_dir_store, name_store):
        with open(f'{results_dir}/results_store.pkl', 'rb') as f:
            results_store = pickle.load(f)
        results_store = {f'{name}_{k}': pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k, v in
                         results_store.items()}
        top = {k: v.iloc[0, :] for k, v in results_store.items()}
        top_results.update(top)
        rmsle_results.update({k: v['mean_test_score'] for k,v in results_store.items()})

    top_results = pd.DataFrame(top_results).T
    top_results['mean_test_score'] *= -1
    rmsle_results = pd.DataFrame(rmsle_results) * -1
    top_results['mean_test_score'].plot.barh()
    plt.show()
    plt.close()
    plt.subplots(figsize=(10, 5))
    sns.boxplot(data=rmsle_results.reset_index().melt(id_vars='index'), x='variable', y='value')
    plt.show(bbox_inches='tight')
    plt.close()
    print('hi')


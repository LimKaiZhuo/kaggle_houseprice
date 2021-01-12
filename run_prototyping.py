import pandas as pd
from own_package.prototyping import lvl1_randomsearch, analyze_xgb_randomsearch

def selector(case):
    if case == 1:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl1_randomsearch(rawdate, './results/lvl1_randomsearch_pp3', preprocess_pipeline_choice=3)
    elif case == 1.1:
        analyze_xgb_randomsearch('./results/lvl1_randomsearch_pp2')


selector(1)


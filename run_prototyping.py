import pandas as pd
from own_package.prototyping import lvl1_randomsearch, analyze_xgb_randomsearch

def selector(case):
    if case == 1:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl1_randomsearch(rawdate, './results/lvl1_randomsearch_pp4', preprocess_pipeline_choice=4)
    elif case == 1.1:
        analyze_xgb_randomsearch('./results/lvl1_randomsearch_pp3')


selector(1)


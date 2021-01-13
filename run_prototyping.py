import pandas as pd
from own_package.prototyping import lvl1_randomsearch, lvl2_ridgecv, lvl2_xgb_randomsearch
from own_package.analysis import analyze_randomsearchresults


def selector(case):
    if case == 1:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl1_randomsearch(rawdate, './results/lvl1_randomsearch_pp5', preprocess_pipeline_choice=5)
    elif case == 1.1:
        analyze_randomsearchresults([f'./results/lvl1_randomsearch_pp{x}' for x in range(1, 5)],
                                 [f'pp{x}' for x in range(1, 5)])
    elif case == 2:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl2_ridgecv(rawdate, './results/lvl2np_ridgecv_pp4', preprocess_pipeline_choice=4,
                     param_dir='./results/lvl1_randomsearch_pp4/results_store.pkl', passthrough=False)
    elif case ==2.1:
        pass
        analyze_randomsearchresults(['./results/lvl2np_ridgecv_pp4'], 'lvl2npRCV')
    elif case == 3:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl2_xgb_randomsearch(rawdate, './results/lvl2np_xgb_pp4', preprocess_pipeline_choice=4,
                     param_dir='./results/lvl1_randomsearch_pp4/results_store.pkl', passthrough=False)


selector(1)

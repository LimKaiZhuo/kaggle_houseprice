import pandas as pd
from own_package.prototyping import lvl1_randomsearch, lvl2_ridgecv
from own_package.analysis import analyze_randomsearchresults


def selector(case):
    if case == 1:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl1_randomsearch(rawdate, './results/lvl1_randomsearch_pp4', preprocess_pipeline_choice=4)
    elif case == 1.1:
        analyze_randomsearchresults([f'./results/lvl1_randomsearch_pp{x}' for x in range(1, 5)],
                                 [f'pp{x}' for x in range(1, 5)])
    elif case == 2:
        rawdate = pd.read_csv('./inputs/train.csv')
        lvl2_ridgecv(rawdate, './results/lvl2np_ridgecv_pp4', preprocess_pipeline_choice=4,
                     param_dir='./results/lvl1_randomsearch_pp4/results_store.pkl', passthrough=False)
        lvl2_ridgecv(rawdate, './results/lvl2pt_ridgecv_pp4', preprocess_pipeline_choice=4,
                     param_dir='./results/lvl1_randomsearch_pp4/results_store.pkl', passthrough=True)
    elif case ==2.1:
        pass
        # analyze_results(['./results/lvl2np_ridgecv_pp4'], 'lvl2npRCV')

selector(2)

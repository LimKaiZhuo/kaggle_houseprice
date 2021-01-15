import pandas as pd
from own_package.prototyping import lvl1_randomsearch, lvl2_ridgecv, lvl2_xgb_randomsearch, lvl2_xgb_vsrandomsearch,\
    lvl1_generate_prediction, lvl2_generate_prediction
from own_package.analysis import analyze_randomsearchresults


def selector(case):
    if case == 1:
        rawdata = pd.read_csv('./inputs/train.csv')
        testdf = pd.read_csv('./inputs/test.csv')
        lvl1_randomsearch(rawdata, testdf, './results/lvl1_randomsearch_pp5', pp_choice=5)
    elif case == 1.1:
        analyze_randomsearchresults([f'./results/lvl1_randomsearch_pp{x}' for x in range(1, 6)],
                                 [f'pp{x}' for x in range(1, 6)])
    elif case == 1.3:
        rawdata = pd.read_csv('./inputs/train.csv')
        testdf = pd.read_csv('./inputs/test.csv')
        lvl1_randomsearch(rawdata, testdf, './results/lvl1_randomsearch_pp5_lt1', pp_choice=5, lt_choice=1)
    elif case == 2:
        rawdata = pd.read_csv('./inputs/train.csv')
        lvl2_ridgecv(rawdata, './results/lvl2pt_ridgecv_pp5', pp_choice=[5, 5, 5],
                     param_dir='./results/lvl1_randomsearch_pp5/results_store.pkl', passthrough=True, final_pp_choice=5)
    elif case ==2.1:
        pass
        analyze_randomsearchresults(['./results/lvl2ptvs_xgb_pp5'], 'lvl2npRCV')
    elif case == 3:
        rawdata = pd.read_csv('./inputs/train.csv')
        lvl2_xgb_randomsearch(rawdata, './results/lvl2pt_xgb_pp5', pp_choice=5,
                              param_dir='./results/lvl1_randomsearch_pp5/results_store.pkl', passthrough=True,
                              final_pp_choice=5)
    elif case == 3.1:
        rawdata = pd.read_csv('./inputs/train.csv')
        lvl2_xgb_vsrandomsearch(rawdata, './results/lvl2ptvs_xgb_pp5', pp_choice=5,
                              param_dir='./results/lvl1_randomsearch_pp5/results_store.pkl', passthrough=True,
                              final_pp_choice=5)
    elif case == 0.1:
        rawdata = pd.read_csv('./inputs/train.csv')
        x_test = pd.read_csv('./inputs/test.csv')
        lvl1_generate_prediction(rawdata, x_test, './results/lvl1_randomsearch_pp1', type_='lvl1_randomsearch',
                                 preprocess_pipeline_choice=1)
    elif case == 0.2:
        rawdata = pd.read_csv('./inputs/train.csv')
        x_test = pd.read_csv('./inputs/test.csv')
        lvl2_generate_prediction(rawdata, x_test, './results/lvl2np_ridgecv_pp4', './results/lvl1_randomsearch_pp4',
                                 type_='lvl2_ridgecv', pp_choice=4)
    elif case == 0.3:
        rawdata = pd.read_csv('./inputs/train.csv')
        x_test = pd.read_csv('./inputs/test.csv')
        lvl2_generate_prediction(rawdata, x_test, './results/lvl2np_xgb_pp4', './results/lvl1_randomsearch_pp4',
                                 type_='lvl2_xgb', pp_choice=4)
    elif case == 0.4:
        rawdata = pd.read_csv('./inputs/train.csv')
        x_test = pd.read_csv('./inputs/test.csv')
        lvl2_generate_prediction(rawdata, x_test, './results/lvl2ptvs_xgb_pp5', './results/lvl1_randomsearch_pp5',
                                 type_='lvl2_xgb', pp_choice=5, passthrough=True, final_pp_choice=5)


selector(1.3)

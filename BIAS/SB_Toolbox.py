from SB_Test_runner import get_test_dict
import numpy as np
import pandas as pd
import pickle
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from statsmodels.stats.multitest import multipletests
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sbs

pwr = importr('PoweR')

#TODO: make sure these directories are correct
def load_ref_vals(n_samples, alpha = 0.01, across = False):
    ''' Load the reference values needed for calculating the p-values
    ----------
    n_samples: the sample size used for the statistical tests. Can only be 
    in [30,50,100,600]
    '''
    if across:
        with open(f"Crit_vals_across/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
            ref_vals, _ = pickle.load(f)
        with open(f"Crit_vals_pwr_across/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
            ref_vals_new, _ = pickle.load(f)
    else:
        with open(f"Crit_vals/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
            _, ref_vals = pickle.load(f)
        with open(f"Crit_vals_pwr/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
            _, ref_vals_new = pickle.load(f)
    return ref_vals, ref_vals_new

p_value_columns = ['1-spacing', '2-spacing', '3-spacing','ad', 'ad_transform', 'shapiro', 'jb', 'ddst']

def get_test_types_new():
    ''' Helper function for the poweR-based tests
    ----------
    '''
    testnames = ['kolmogorov',
 'CvM',
 'AD_pwr',
 'Durbin',
 'Kuiper',
 'HG1',
 'HG2',
 'Greenwood',
 'QM',
 'RC',
 'Moran',
 'Cressie1',
 'Cressie2',
 'Vasicek',
 'Swartz',
 'Morales',
 'Pardo',
 'Marhuenda',
 'Zhang1',
 'Zhang2']
    test_types_new = [pwr.create_alter(robjects.FloatVector(np.arange(63,83)))[i][0] for i in range(20)]
    return {k:v for k,v in zip(testnames, test_types_new)}

def transform_to_reject_dt_corr(dt, alpha, n_samples, correction_method='fdr_bh'):
    ''' Apply p-value corrections on the dataframe of test statistics
    ----------
    dt: The DataFrame containing the calculated test statistics for each dimension
    alpha: The threshold for statistical significance
    n_samples: the sample size used for the statistical tests. Can only be 
    in [30,50,100,600]
    correction_method: Which type of p-value correction to apply. Recommended is 'fdr_bh', 
    but 'fdr_by' and 'holm' are also supported.
    '''
    reference_vals, ref_vals_new = load_ref_vals(n_samples)
    test_types_new = get_test_types_new()    
    
    dt_rejections = pd.DataFrame()
    dt_p_vals_temp = pd.DataFrame()
    for colname in p_value_columns:
        dt_rejections[colname] = multipletests(dt[colname], alpha=alpha, method=correction_method)[0]
        
    for k,v in reference_vals.items():
        if 'kurt' in k:
            temp = [percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
            temp = [min(x, 1-x) for x in temp] #two-sided comparison
            dt_rejections[k] = multipletests(temp, alpha=alpha/2, method=correction_method)[0]
        elif k in ['min', 'wasserstein', 'mdd_max', 'mdd_min']:
            temp = [1-percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
            dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
        else:
            temp = [percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
            dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
    for k,v in ref_vals_new.items():
        if test_types_new[k] == 4:
            temp = [percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
            dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
        else:
            temp = [1-percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
            dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
    return dt_rejections

def get_test_names_dict():
    ''' Helper function to ensure consistent naming for the used statistical tests
    by creating a dictionary
    ----------
    '''
    test_dict_per = get_test_dict(n_samples=100, per_dim=True)
    test_names = list(test_dict_per.keys())
    test_names.remove('AD_pwr')
    test_names_paper = ['1-spacing',
     '2-spacing',
     '3-spacing',
     'range',
     'min',
     'max',
     'AD',
     'tAD',
     'Shapiro',
     'JB',
     'LD-min',
     'LD-max',
     'Kurt',
     'MPD-max',
     'MPD-min',
     'Wasserstein',
     'NS',
     'KS',
     'CvM',
     'Durbin',
     'Kuiper',
     'HG1',
     'HG2',
     'Greenwood',
     'QM',
     'RC',
     'Moran',
     'Cressie1',
     'Cressie2',
     'Vasicek',
     'Swartz',
     'Morales',
     'Pardo',
     'Marhuenda',
     'Zhang1',
     'Zhang2']

    test_label_dict = {k:v for k,v in zip(test_names, test_names_paper)}
    return test_label_dict


def plot_swarm_with_heatmap(data, rejections, filename = None):
    ''' Plotting function to create the swarmplot + rejection heatmap
    ----------
    data: The DataFrame containing the final position values
    rejections: The DataFrame containing the corresponding test rejections
    filename: If not none, the name of the file to store the figure
    '''
    #     test_dict_per = get_test_dict(n_samples=100, per_dim=True) #Note: n_samples doesn't impact, just need the keys
    test_label_dict = get_test_names_dict()
    data_dt = pd.DataFrame(data)
    fig, axs = plt.subplots(2, figsize=(19,14), sharex=True)
    ax1 = axs[0]
    dt_molt = data_dt.melt()
    dt_molt['variable'] = dt_molt['variable'] + 1.5
    sbs.swarmplot(data=dt_molt, x='variable', y='value', ax=ax1)
    ax1.set_xlim(-0.5, 29.5)
    for dim in range(30):
        c0 = ax1.get_children()[dim]
        c0.set_offsets([[x+0.5,y] for x,y in c0.get_offsets()])
        ax1.axvline(dim, color='k', lw=0.6, ls=':')
    sbs.heatmap(np.array(rejections).transpose(), ax=axs[1], cbar=False, 
                yticklabels=[test_label_dict[x] for x in rejections.columns], linewidths=.01, cmap='crest_r')

    ax1.set_xlabel("")
    axs[1].set_xlabel("Dimension", fontsize=16)
    axs[1].set_xticklabels(range(1,31), fontsize=14)
    axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=14)
    ax1.set_ylabel("Value", fontsize=16)
    ax1.set_ylim(0,1)
    ax1.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=14)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
        
def f0(x):
    return np.random.uniform()
        
        
def predict_type(dt_rej,  print_type = False):
    mean_rej = (np.mean(np.array(dt_rej), axis=0) >= 0.1)
    if np.sum(mean_rej) == 0:
        if print_type:
            print('No clear evidence of bias detected')
        return 'none'
    
    with open("RF_few_classes.pkl", "rb") as input_file:
        rf = pickle.load(input_file)
    res_class = rf.predict(mean_rej.reshape(1, -1))
    classes = rf.classes_
    prob_classes = rf.predict_proba(mean_rej.reshape(1, -1))
    
    with open("RF_rejection_based.pkl", "rb") as input_file:
        rf = pickle.load(input_file)
    res_scen = rf.predict(mean_rej.reshape(1, -1))
    scennames = rf.classes_
    prob_scens = rf.predict_proba(mean_rej.reshape(1, -1))
    
    if print_type:
        print(f"Detected bias which seems to be related to {res_class} ({np.max(prob_classes):.2f} probability)." +
              f"The rejections seems to be most similar to the {res_scen} scenario ({np.max(prob_scens):.2f} probability).")
    return {'Class' : res_class[0], 'Class Probabilities' : prob_classes, 
            'Scenario' : res_scen[0], 'Scenario Probabilities' : prob_scens}
    
def run_SB_test(data, corr_method = 'fdr_by', alpha=0.01, show_figure=False, filename = None, print_type = True):
    ''' The main function used to detect Structural Bias
    ----------
    data: The matrix containing the final position values on F0. Note that these should be scaled 
    in [0,1], and in the shape (n_samples, dimension), where n_samples is in [30, 50, 100, 600] 
    alpha: The threshold for statistical significance
    correction_method: Which type of p-value correction to apply. Recommended is 'fdr_bh', 
    but 'fdr_by' and 'holm' are also supported.
    show_figure: Whether or not to create a plot of the final positions and the corresponding test rejections
    filename: If not none, the name of the file to store the figure (only when show_figure is True)
    print_type: Wheter or not to print the predicted type of SB
    '''
    DIM = data.shape[1]
    n_samples = data.shape[0]
    if not n_samples in [30,50,100,600]:
        raise ValueError("Sample size is not supported")
    if print_type:
        print(f"Running SB calculation with {DIM}-dimensional data of sample size {n_samples} (alpha = {alpha})")
    records = {}    
    test_battery_per_dim = get_test_dict(n_samples)
    for tname, tfunc in test_battery_per_dim.items():
        temp = []
        for r in range(DIM):
            try:
                temp.append(tfunc(data[:,r], alpha=alpha))
            except:
                next
        records[tname] = temp
    dt = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in records.items() ]))
    dt_rejections = transform_to_reject_dt_corr(dt, alpha, n_samples, corr_method)
    #Drop duplicate test
    dt_rejections = dt_rejections.drop('AD_pwr', axis=1)
    
    if show_figure:
        plot_swarm_with_heatmap(data, dt_rejections, filename)
    
    return dt_rejections, predict_type(dt_rejections, print_type)


def transform_to_reject_dt_across(dt, alpha, n_samples):
    crit_vals, crit_vals_new = load_ref_vals(n_samples, alpha, True)
    test_types_new = get_test_types_new()    
    
    dt_rejections = pd.DataFrame()
    for colname in p_value_columns:
        dt_rejections[colname] = dt[colname] < alpha
        
    #Ugly solution to distinguish two-sided vs one-sided tests
    dt_rejections['kurtosis'] = (crit_vals['kurtosis_low'] > dt['kurtosis']) | (dt['kurtosis'] > crit_vals['kurtosis_high'])
    dt_rejections['mmpd'] = (crit_vals['mmpd_low'] > dt['mmpd']) | (dt['mmpd'] > crit_vals['mmpd_high'])
    dt_rejections['mi'] = (crit_vals['mi_low'] > dt['mi']) | (dt['mi'] > crit_vals['mi_high'])
    dt_rejections['med_ddlud'] = (crit_vals['med_ddlud_low'] > dt['med_ddlud']) | (dt['med_ddlud'] > crit_vals['med_ddlud_high'])
    for k,v in crit_vals.items():
        if 'kurt' in k or 'low' in k or 'high' in k:
            next
        else:
            if k in ['max_ddlud']:
                dt_rejections[k] = dt[k] > v
            else:
                dt_rejections[k] = dt[k] < v
                
    for k,v in crit_vals_new.items():
        if test_types_new[k] == 4:
            dt_rejections[k] = dt[k] < v
        else:
            dt_rejections[k] = dt[k] > v
    return dt_rejections

def run_multidim_sb_tests(data, alpha=0.01, print_type = True):
    DIM = data.shape[1]
    n_samples = data.shape[0]
    if not n_samples in [30,50,100,600]:
        raise ValueError("Sample size is not supported")
    if DIM != 30:
        raise ValueError("Only 30-dimensional data is supported for across-dimension testing")
    if print_type:
        print(f"Running SB calculation with {DIM}-dimensional data of sample size {n_samples} (alpha = {alpha})")
    records = {}    
    test_battery_across_dim = get_test_dict(n_samples, per_dim=False)
    for tname, tfunc in test_battery_across_dim.items():
        try:
            records[tname] = tfunc(data)
        except:
            next
    #TODO: fix this function
    dt = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in records.items() ]))
    dt_rejections = transform_to_reject_dt_across(dt, alpha, n_samples)
    failed_tests = [x for x in dt_rejections.columns if np.sum(dt_rejections[x]) > 0 ]
    if print_type:
        if len(failed_tests == 0):
            print('No clear evidence of bias detected')
        else:
            print(f'The following tests detected potential structural bias: {failed_tests}')
    return failed_tests
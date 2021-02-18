import numpy as np
from scipy.stats import chi2

def likelihood_ratio_test(base_model, alt_model, verbose=True):
    '''
    
    Likelihood ratio test for comparison of nested models. 
    
    '''
    
    df = alt_model.df_model - base_model.df_model
    LR = 2*(alt_model.llf - base_model.llf)
    p_value = chi2.sf(LR, df)
    
    if verbose:
        print('Likelihood ratio: %.3f'%LR)
        print('Degrees of freedom: %d'%df)
        print('p-value: ' + str(p_value))
        
    return p_value, LR, df

def ismember(curr_list, check_list):
    '''
    Check if curr_list contains any overlapping value with check_list.
    Returns a list of booleans with same size as curr_list.
    
    '''
    
    return [value in check_list for value in curr_list]

def nan_reject(array_list, match_list=False):
    '''
    Reject NaNs from an array or a list of array.
    If match_list=True, arrays in the list are treated as columns of a 2D array. 
    
    '''
    
    def nan_reject_array(array):
        if len(np.shape(array))==1:
            rej_idx = np.logical_not(np.isnan(array))
        else:
            rej_idx = np.logical_not(np.sum(np.isnan(array), axis=1))
        return array[rej_idx], rej_idx
    
    if type(array_list) is np.ndarray:
        return nan_reject_array(array_list)[0]
    elif type(array_list) is list:
        if match_list:
            rej_idx_all = np.column_stack([nan_reject_array(a)[1] for a in array_list])
            rej_idx = np.sum(rej_idx_all, axis=1) == len(array_list)
            return [array[rej_idx] for array in array_list]
        else:
            return [nan_reject_array(array)[0] for array in array_list]
    
def bootstrap(inputs, function, nperm, alpha=0.05):
    '''
    
    Perform bootstrapping on function(*input), with nperm permutations.
    Returns the list of bootstrapped values, as well as mean and confidence intervals
    (default: 95% CI)
    
    '''
    
    m = np.shape(inputs[0])[0]
    stat = np.zeros(nperm)*np.nan
    for k in range(nperm):
        idx_vec = np.random.choice(m, size=m, replace=True)
        stat[k] = function(*[x[idx_vec] for x in inputs])[0]

    return stat, np.mean(stat), np.array([np.percentile(stat,100*alpha/2), np.percentile(stat,100*(1-alpha/2))])
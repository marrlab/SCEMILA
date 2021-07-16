import random as r

'''Script splits data into x folds, and returns the specific sets
with the function return_folds'''

data_split = None

def split_in_folds(data, num_folds): 
    '''splits data into num_folds shares. Split
    data can then be retrieved using return_folds.
    Formats the data passed to the function and splits 
    it into folds which can then be accessed easily

    Accepts:
    - data (dictionary):
        Data in a dictionary for stratified split:
        data[bag label] = [list of patients]
        The function stratifies the data based on
        the bag label.
    - num_folds (integer):
        number of folds to split data into. 
    
    Returns:
    - Nothing. Access folds through the return_folds method.
    '''
    
    global data_split

    data_split = dict()
    percent_per_split = 1/num_folds

    # iterate over all entities
    for key, value in data.items():

        # following routine makes sure data is always split in the same way
        r.seed(42)
        ordered_patients = value
        ordered_patients.sort()
        r.shuffle(ordered_patients)

        # output data for every fold
        for fold in range(num_folds):
            if not fold in data_split:
                data_split[fold] = dict()
            
            # calculate starting and ending idx for list of patients
            start = round(fold*percent_per_split*len(ordered_patients))
            end = round((fold+1)*percent_per_split*len(ordered_patients))

            data_split[fold][key] = ordered_patients[start:end]

def return_folds(folds):

    '''Returns all data from data_split from the corresponding folds of the 
    previously calculated split.
    
    Accepts:
    - folds (integer or list of integers):
        Pass the amount of folds that should be retrieved.
    
    Returns:
    - data_final (dictionary):
        dictionary of patients, containing all the data from the requested folds.
        The data is formatted as
        
        data_final[bag label] = [list of patients]
    '''

    if(isinstance(folds, int)):
        folds = [folds]

    data_final = dict()
    
    # merge together multiple folds to return one dictionary
    for fold in folds:
        for key, value in data_split[fold].items():

            if not key in data_final:
                data_final[key] = []
            data_final[key].extend(value)

    return data_final
    



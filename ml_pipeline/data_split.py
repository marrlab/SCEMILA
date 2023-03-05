import random as r

'''Script splits data into x folds, and returns the specific sets
with the function return_folds'''

data_split = None


def split_in_folds(data, num_folds):
    '''splits data into num_folds shares. Split
    data can then be retrieved using return_folds.
    Data comes in a dict which has to be formatted as:

        data[entity] = [patient_list]

    So that in the end every fold of num_folds contains a
    stratified part of patients for every entity.
    '''

    global data_split

    data_split = dict()
    percent_per_split = 1 / num_folds

    # iterate over all entities
    for key, value in data.items():

        # following routine makes sure data is always split in the same way
        r.seed(42)
        ordered_patients = sorted(value)
        r.shuffle(ordered_patients)

        # output data for every fold
        for fold in range(num_folds):
            if fold not in data_split:
                data_split[fold] = dict()

            # calculate starting and ending idx for list of patients
            start = round(fold * percent_per_split * len(ordered_patients))
            end = round((fold + 1) * percent_per_split * len(ordered_patients))

            data_split[fold][key] = ordered_patients[start:end]


def return_folds(folds):
    '''Returns all data from data_split from the corresponding folds of the
    previously calculated split, can pass either a single integer or a list of
    integers for the folds to fetch.'''

    if(isinstance(folds, int)):
        folds = [folds]

    data_final = dict()

    # merge together multiple folds to return one dictionary
    for fold in folds:
        for key, value in data_split[fold].items():

            if key not in data_final:
                data_final[key] = []
            data_final[key].extend(value)

    return data_final

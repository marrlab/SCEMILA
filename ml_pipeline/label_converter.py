import pandas as pd


'''
In the following, The following terms are used:
- true label = string label of our class
- artificial label = integer label for one-hot encoding
'''


class LabelConverter:
    '''Keeps track of all true labels and allows for easy index conversion.
    Instead of using a fixed dictionary, this allows us to easily adapt to different
    classification scenarios, if e.g. changing code in the dataset.process_label function'''

    def __init__(self, exclude_lbls=[], path_preload=None):
        '''Initialize. 

        Accepts:
        - exclude_lbls (List of Strings):
            Ignore all of the labels passed here.
        - path_preload (path):
            Load pre-existing dataframe for the label conversion

        Returns:
        - Nothing
        '''

        if not path_preload is None:
            self.df = pd.read_csv(path_preload, index_col=0)
            if(self.df.index.name == 'art_lbl'):
                self.df = self.df.reset_index()

        else:
            self.df = pd.DataFrame([], columns=['art_lbl', 'true_lbl',
                                                's_train', 's_val', 's_test', 'size_tot'])

        self.exclude = exclude_lbls

    def add(self, true_lbl, size=1, split=None):
        '''Add a new entry. If label already exists, just increase the class size count.
        If new entry, automatically matches next free integer to class.

        Accepts:
        - true_lbl (String):
            String label of the patient added. 
        - size (Integer):
            Amount of patients added with this label.
        - split (String):
            Part of the dataset the patient belongs to, e.g.
            "train", "test", or "val". Like this, we keep track
            of the amount of patients in each part of the dataset.

        Returns:
        - Nothing
        '''

        # exclude sample, if in exclusion list
        if true_lbl in self.exclude:
            return

        # introduce new row if true label is not yet contained
        if not true_lbl in self.df['true_lbl'].values:
            self.df.loc[len(self.df)] = (len(self.df), true_lbl, 0, 0, 0, 0)

        # add class size information
        if split in ['train', 'test', 'val']:
            s_increase = 's_' + split
            self.df.loc[self.df['true_lbl'] == true_lbl, s_increase] += size
        self.df.loc[self.df['true_lbl'] == true_lbl, 'size_tot'] += size

    def __getitem__(self, input):
        '''easy access: checks if input is int --> convert to true label, or 
        input is string --> convert to artificial label.

        Accepts:
        - input (String or Integer)
            Function figures out, which way the conversion should take place, 
            then calls the proper conversion.

        Returns:
        - Converted label (Integer or String)
        '''

        if isinstance(input, int):
            conv_from = 'art_lbl'
            conv_to = 'true_lbl'
        else:
            conv_from = 'true_lbl'
            conv_to = 'art_lbl'

        return self.convert(input, conv_from, conv_to)

    def convert(self, input, conv_from, conv_to):
        '''Convert label using dataframe'''

        if not (self.df[conv_from] == input).any():
            raise NameError(
                'Trying to convert value which does not exist in label_converter!')

        return self.df.loc[self.df[conv_from] == input, conv_to].item()

    def get_sizes(self, label):
        '''return size of a class with string label label

        Accepts:
        - label (String):
            Label, where class size is asked

        Returns:
        - pd.Series:
            Amount of patients in different folds
        '''

        if isinstance(label, int):
            label = self[label]

        if not (self.df['true_lbl'] == label).any():
            raise NameError('Trying to access size of unknown label!')

        columns = ['s_train', 's_val', 's_test', 'size_tot']
        return self.df.loc[self.df['true_lbl'] == label, columns].values[0]

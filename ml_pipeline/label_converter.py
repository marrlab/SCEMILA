import pandas as pd


'''
In the following, I refer to the two terms:
- true label = string label of our class
- artificial label = integer label for one-hot encoding
'''


class LabelConverter:
    '''Keeps track of all true labels and allows for easy index conversion.
    This is required since every class needs to match one corresponding integer
    for one-hot encoding. The dataframe can be integrates with confusion
    matrix function.'''

    def __init__(self, exclude_lbls=[], path_preload=None):
        '''Initialize. Parameters:
        - exclude_lbls: can put in a list of labels which will be discarded
        - path_preload: if a path is put here, load dataframe instead of setting
        up a new one'''

        if path_preload is not None:
            self.df = pd.read_csv(path_preload, index_col=0)
            if(self.df.index.name == 'art_lbl'):
                self.df = self.df.reset_index()

        else:
            self.df = pd.DataFrame(
                [],
                columns=[
                    'art_lbl',
                    'true_lbl',
                    's_train',
                    's_val',
                    's_test',
                    'size_tot'])

        self.exclude = exclude_lbls

    def add(self, true_lbl, size=1, split=None):
        '''Add a new entry. If label already exists, just counts the class size.
        If new entry, automatically matches next free integer to class.
        - true_lbl: string label of class
        - size: amount of entries to add
        - split: fold, to which the added entries belong'''

        # exclude sample, if in exclusion list
        if true_lbl in self.exclude:
            return

        # introduce new row if true label is not yet contained
        if true_lbl not in self.df['true_lbl'].values:
            self.df.loc[len(self.df)] = (len(self.df), true_lbl, 0, 0, 0, 0)

        # add class size information
        if split in ['train', 'test', 'val']:
            s_increase = 's_' + split
        self.df.loc[self.df['true_lbl'] == true_lbl, s_increase] += size
        self.df.loc[self.df['true_lbl'] == true_lbl, 'size_tot'] += size

    def __getitem__(self, input):
        '''easy access: checks if input is int --> convert to true label, or
        input is string --> convert to artificial label.'''

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
        '''return size of a class with string label label'''

        if isinstance(label, int):
            label = self[label]

        if not (self.df['true_lbl'] == label).any():
            raise NameError('Trying to access size of unknown label!')

        columns = ['s_train', 's_val', 's_test', 'size_tot']
        return self.df.loc[self.df['true_lbl'] == label, columns].values[0]

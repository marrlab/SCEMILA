import sys, os, time
import argparse as ap

# strange workaround for bug: libgcc_s.so.1 must be installed for pthread_cancel to work
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
torch.multiprocessing.set_sharing_strategy('file_system')

# import from other, own modules
import label_converter      # makes conversion from string label to one-hot encoding easier
from dataset import *       # dataset
from model import *         # actual MIL model
from model_train import *   # model training function


# 1: Setup. Source Folder is parent folder for both mll_data_master and the /data folder
TARGET_FOLDER = '/storage/groups/qscd01/projects/aml_mil_hehr/final_results/testing_stability/'       # results will be stored here
SOURCE_FOLDER = '/storage/groups/qscd01/datasets/210526_mll_mil_pseudonymized/'                             # path to dataset

# get arguments from parser, set up folder
##### parse arguments
parser = ap.ArgumentParser()

########## Algorithm / training parameters
parser.add_argument('--fold', help='offset for cross-validation (1-5). Change to cross-validate', required=False, default=0)# shift folds for cross validation. Increasing by 1 moves all folds by 1.
parser.add_argument('--lr', help='used learning rate', required=False, default=0.00005)                                     # learning rate
parser.add_argument('--ep', help='max. amount after which training should stop', required=False, default=150)               # epochs to train
parser.add_argument('--es', help='early stopping if no decrease in loss for x epochs', required=False, default=20)          # epochs without improvement, after which training should stop.
parser.add_argument('--multi_att', help='use multi-attention approach', required=False, default=1)                          # use multiple attention values if 1

########## Data parameters: Modify the dataset
parser.add_argument('--prefix', help='define which set of features shall be used', required=False, default='fnl34_')        # define feature source to use (from different CNNs)
parser.add_argument('--filter_diff', help='Filters AML patients with less than this perc. of MYB.', default=20)             # pass -1, if no filtering acc to peripheral blood differential count should be done
parser.add_argument('--filter_mediocre_quality', help='Filters patients with sub-standard sample quality', default=1)       # Leave out some more samples, if we have enough without them. Quality of these is not good, but if data is short, still ok.
parser.add_argument('--bootstrap_idx', help='Remove one specific patient at pos X', default=-1)                              # Remove specific patient to see effect on classification

########## Output parameters
parser.add_argument('--result_folder', help='store folder with custom name', required=True)                                 # custom output folder name
parser.add_argument('--save_model', help='choose wether model should be saved', required=False, default=0)                  # store model parameters if 1
args = parser.parse_args()

# store results in target folder
TARGET_FOLDER = os.path.join(TARGET_FOLDER, args.result_folder)
if not os.path.exists(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)
start = time.time()





# 2: Dataset
# Initialize datasets, dataloaders, ...
print("")
print('Initialize datasets...')
label_conv_obj = label_converter.label_converter()
set_dataset_path(SOURCE_FOLDER)
define_dataset(num_folds = 5, prefix_in = args.prefix, 
                label_converter_in=label_conv_obj, filter_diff_count=int(args.filter_diff), 
                filter_quality_minor_assessment=int(args.filter_mediocre_quality))
datasets = {}

##### set up folds for cross validation
folds = {'train':np.array([0,1,2]), 'val':np.array([3]), 'test':np.array([4])}
for name, fold in folds.items():
    folds[name] = ((fold+int(args.fold))%5).tolist()

datasets['train'] = dataset(folds=folds['train'], aug_im_order=True, split='train', patient_bootstrap_exclude=int(args.bootstrap_idx))
datasets['val'] = dataset(folds=folds['val'], aug_im_order=False, split='val')
datasets['test'] = dataset(folds=folds['test'], aug_im_order=False, split='test')

##### store conversion from true string labels to artificial numbers for one-hot encoding
df = label_conv_obj.df
df.to_csv(os.path.join(TARGET_FOLDER, "class_conversion.csv"), index=False)
class_count = len(df)
print("Data distribution: ")
print(df)

##### Initialize dataloaders
print("Initialize dataloaders...")
dataloaders= {}

# ensure balanced sampling
class_sizes = list(df.size_tot)                                                             # get total sample sizes
label_freq = [class_sizes[c]/sum(class_sizes) for c in range(class_count)]                  # calculate label frequencies
individual_sampling_prob = [(1/class_count)*(1/label_freq[c]) for c in range(class_count)]  # balance sampling frequencies for equal sampling

idx_sampling_freq_train = torch.tensor(individual_sampling_prob)[datasets['train'].labels]
idx_sampling_freq_val = torch.tensor(individual_sampling_prob)[datasets['val'].labels]

sampler_train = WeightedRandomSampler(weights=idx_sampling_freq_train, replacement=True, num_samples=len(idx_sampling_freq_train))
# sampler_val = WeightedRandomSampler(weights=idx_sampling_freq_val, replacement=True, num_samples=len(idx_sampling_freq_val))

dataloaders['train'] = DataLoader(datasets['train'], num_workers=1, sampler=sampler_train)
dataloaders['val'] = DataLoader(datasets['val'], num_workers=1)#, sampler=sampler_val)
dataloaders['test'] = DataLoader(datasets['test'], num_workers=1)
print("")





# 3: Model
# initialize model, GPU link, training

##### set up GPU link and model (check for multi GPU setup)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()
print("Found device: ", ngpu, "x ", device)

model = AMiL(class_count=class_count, multicolumn=int(args.multi_att), device=device)

if(ngpu > 1):
    model = torch.nn.DataParallel(model)
model = model.to(device)
print("Setup complete.")
print("")

##### set up optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.9, nesterov=True)
scheduler = None

##### launch training
train_obj = trainer(model=model, dataloaders=dataloaders, epochs=int(args.ep), optimizer = optimizer,
                    scheduler=scheduler, class_count=class_count, early_stop=int(args.es), device=device)
model, conf_matrix, data_obj = train_obj.launch_training()




# 4: aftermath
# save confusion matrix from test set, all the data , model, print parameters

np.save(os.path.join(TARGET_FOLDER, 'test_conf_matrix.npy'), conf_matrix)
pickle.dump(data_obj, open(os.path.join(TARGET_FOLDER, 'testing_data.pkl'), "wb"))

if(int(args.save_model)):
    torch.save(model, os.path.join(TARGET_FOLDER, 'model.pt'))
    torch.save(model, os.path.join(TARGET_FOLDER, 'state_dictmodel.pt'))

end = time.time()
runtime = end-start
time_str = str(int(runtime//3600)) + "h" + str(int((runtime%3600)//60)) + "min" + str(int(runtime%60)) + "s"

##### other parameters
print("")
print("------------------------Final report--------------------------")
print('prefix', args.prefix)
print('Runtime', time_str)
print('max. Epochs', args.ep)
print('Learning rate', args.lr)


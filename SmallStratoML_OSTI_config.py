
from NNfunctions import *

### OSTI ### CONFIG

BSP = "SmallStratoML"

t0 = 0.0                                                                # time configuration, set beginning, end and number of relevant time steps
tf = 172799.0
tf = 86399.0
nTimes = 25

timepoints = np.linspace(t0,tf,num=nTimes)                         # create time points array

nepoch = 100                                                        # number of training epochs

nsamples = 390                                                          # number of data samples to use (data has to exist!)
val_perc = 0.2
test_perc = 0.02
 
#nlayers = 3                                                             # number and sizes of hidden layers
#hidden_sizes = [150 for i in range(nlayers)]
hidden_sizes = [ 30, 130, 25 ]

emisnames_in = np.array([])                                             # emisnames_in: names of emissions given into NN
spcnames     = np.array(["O1D","O","O3","O2","NO","NO2"])               # spcnames_out: names of species to predict

met_names = np.array(["time"])

#percentage to identify outliers in loss tracking (gets applied twice, for lowest and highest)
outlier_perc = 0.05

# normalization function
F_normal = norm_arb2
norm_cutperc = 0.05

learning_rate  = 0.02
momentum = 0.9
learning_gamma = 0.3
patience = 5
threshold = 0.05

# turn actual training on and off
train = False
train = True


from IMPORTS import *
from NNfunctions import *

BSP = "RACM_NN"
# insert path with matrix files
sys.path.insert(1, 'MechanismMeta/'+BSP)

device1 = torch.device('cuda:1')
#device1 = torch.device('cpu')

##################################################################

# # # # # # # # # #   IMPORT CHININ MATRICES   # # # # # # # # # #

##################################################################

# import A matrix (educt stoichiometry)
# name of imported variable: A
from A_matrix import *

# import D matrix (D=(B-A)^T)
# name of imported variable: D
from D_matrix import *

#for RACM: delete last two reactions and last 3 species (these are non-gaseous)
A = A[:286,:95]
D = D[:95,:286]

# create E matrix (contains instructions to multiply educt concentrations)
# E[i] = list of spc ids that are educt in reaction i
E=[]
for i in range(A.shape[0]):
    E.append(np.nonzero(A[i,:])[0])


##################################################################

# # # # # # # # # # # #   FEATURE NAMES   # # # # # # # # # # # #

##################################################################

# import ordered species names (ordered like A and D)
from spcnames_order import *
spcids = {}
for (id, name) in enumerate(spcnames):
    spcids[name] = id

spcnames = spcnames[:95]

emisnames = np.array(["NO", "ISO", "API", "ETE", "HC5"])

metnames = np.array(["time"])

spcnames_imp = np.array([
    'O3',
    'NO2',
    'GLY',
    'MGLY',
    'ORA2',
    'HCHO',
    'HO2',
    'HO',
    'H2O2',
    'ORA1'])


#spcnames_imp = np.array([
#    'O3',
#    'NO2',
#    'HCHO',
#    'HO2',
#    'HO',
#    'H2O2'])


spcids_imp = np.zeros((spcnames_imp.size))
for (i, name) in enumerate(spcnames_imp):
    spcids_imp[i] = spcids[name]

spcnames_plot = spcnames_imp



##################################################################

# # # # # # # # # # # #   DATA SET SPECS   # # # # # # # # # # # #

##################################################################



# zero concentrations at t=0 are "unrealistic", we take pre-simulated data
t_0 = 0.0+60*60                                                                # time configuration, time range of the data to read out of files
t_final = 86399.0+60*60

NN_dt = 60*60 # in seconds!                                                 # timestep the NN predicts

nTimes = int(round( (t_final-t_0) / NN_dt ,0)) + 1                                 # predictions per data set time span (which is [t_0, t_final] )

timepoints = np.linspace(t_0,t_final,num=nTimes)                         # create time points array

nFiles = 729                                                          # number of data samples to use (data has to exist!)
val_perc = 0.1                                                        
test_perc = 0.01



##################################################################

# # # # # # # # # # # # #   NN SPECS   # # # # # # # # # # # # #

##################################################################


nepoch = 15                                                        # number of training epochs

# specifiy sizes of hidden layers
#hidden_sizes = [100 for i in range(nlayers)]
hidden_sizes = [ 40,300,300 ]
#hidden_sizes = [ 40,300 ]

# RESNET
#ls=250
#hidden_sizes = [ [ls],\
#                 [ls] ]
#n_encoded = 40


#percentage to identify outliers in loss tracking (gets applied twice, for lowest and highest)
outlier_perc = 0.1

# Scaling: MinMax or log
Scaling = 'log'                                                      # data scaling (pre-processing)
if Scaling=='log':
    met_scaling=False
else:
    met_scaling=True


## LOG SCALER parameters
if Scaling=='log':
    # single training parameters
    learning_rate_s  = 0.0002
    momentum_s = 0.9
    learning_gamma_s = 0.25
    patience_s = 5
    threshold_s = 0.1

    # diurnal training parameters
    learning_rate  = 0.1
    momentum = 0.9
    learning_gamma = 0.1
    patience = 5
    threshold = 0.1

## MINMAX SCALER parameters
if Scaling=='MinMax':
    # single training parameters
    #learning_rate_s  = 0.1
    learning_rate_s  = 0.02
    momentum_s = 0.9
    learning_gamma_s = 0.25
    patience_s = 5
    threshold_s = 0.1
    
    # diurnal training parameters
    learning_rate  = 0.1
    momentum = 0.9
    learning_gamma = 0.1
    patience = 5
    threshold = 0.1


##################################################################

# # # # # # # # # # # # #   SWITCHES   # # # # # # # # # # # # #

##################################################################


# turn actual training on and off
train_single = True
train_diurnal = False

# Plotting on and off switch
Plotting = True


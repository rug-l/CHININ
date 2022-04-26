from IMPORTS import *

# #################################################### #
#                                                      #
#                      FUNCTIONS                       #
#                                                      #
# #################################################### #

def identity(x,dim=1):
    return x

def norm_arb(x):
    normedx = x/max(abs(x))
    return normedx

def minmax_scaler(x,datmin,datmax):
    factor=1
    if datmax-datmin<1E-16:
        return x*0
    normedx = (x-datmin)/(datmax-datmin+datmin*1E-16)
    return normedx*factor

def log_scaler(x,datmin,datmax):
    if datmax-datmin<1E-16:
        return 0
    normedx = np.log(np.maximum(x,np.ones_like(x)))
    return normedx


def find_dat_minmax(ini_in, emis_in, spc_out, data_tr, target_tr, data_val, target_val, data_te, target_te, nPosTimes, cut_perc = 0.0):
    dat_minmax = {}
    for i, spc in enumerate(spc_out):
        spc_values = np.concatenate((target_tr[:,i*nPosTimes:(i+1)*nPosTimes], target_val[:,i*nPosTimes:(i+1)*nPosTimes], target_te[:,i*nPosTimes:(i+1)*nPosTimes]))
        spc_values = np.sort(spc_values.flatten())
        n_values = spc_values.size
        spc_values = spc_values[int(cut_perc*n_values):int((1-cut_perc)*n_values)]
        dat_minmax[spc] = [spc_values[0], spc_values[-1]]
    #print(spc+":")
    #print("%.6e" % spc_values[0])
    #print("%.6e" % spc_values[-1])

    for i,spc in enumerate(ini_in):
        spc_values = np.concatenate((data_tr[:,i], data_val[:,i], data_te[:,i]))
        spc_values = np.sort(spc_values.flatten())
        n_values = spc_values.size
        spc_values = spc_values[int(cut_perc*n_values):int((1-cut_perc)*n_values)]
        if np.any(spc_out == spc):
            dat_minmax[spc] = [min(dat_minmax[spc][0], spc_values[0]), max(dat_minmax[spc][1], spc_values[-1])]
        else:
            dat_minmax[spc] = [spc_values[0], spc_values[-1]]

    for i,spc in enumerate(emis_in, ini_in.size):
        spc_values = np.concatenate((data_tr[:,i], data_val[:,i], data_te[:,i]))
        spc_values = np.sort(spc_values.flatten())
        n_values = spc_values.size
        spc_values = spc_values[int(cut_perc*n_values):int((1-cut_perc)*n_values)]
        # CAUTION: trouble if some emis is both given in and target, because concentration has different magnitude than emission rate! 
        if np.any(spc_out == spc) or np.any(ini_in == spc):
            dat_minmax[spc] = [min(dat_minmax[spc][0], spc_values[0]), max(dat_minmax[spc][1], spc_values[-1])]
        else:
            dat_minmax[spc] = [spc_values[0], spc_values[-1]]

    return dat_minmax

def findConfigValueStr(configfile, parameter):
    value_str = None
    for line in fileinput.input(file, inplace=1):
        if parameter in line:
            value_str = line[line.find("=")+1:]
        sys.stdout.write(line)
    if value_str is None:
        print("Couldn\'t find configuration value \""+parameter+"\". Quit.")
        sys.exit()
    return value_str

def fixTimesteps(y, t, t_desired):
    """Approximate desired time steps with linear functions in between of given time steps.
       NOTE: It is important to have the last entry of t exceeding the last entry of t_desired!"""
    y_data = torch.zeros(t_desired.size)
    # first entry seperately
    if abs(t[0]-t_desired[0])<1E-12:
        y_data[0] = y[0]
        start_inner=1
    else:
        start_inner=0
    # loop inner entries
    for i in range(start_inner,y_data.size(dim=0)):
        u=firstgreaterentry(t,t_desired[i])
        l=u-1
        y_data[i] = y[l]+((y[u]-y[l])/(t[u]-t[l]))*(t_desired[i]-t[l])
    
    return y_data

def DataFromNetCDF(BSP, sample, ininames_in, spcnames_out, met_names, timepoints):
    fn = "../../AtCSol/NetCDF/MLData/"+BSP+"/"+BSP+"_"+str(sample)+".nc"
    ds=nc.Dataset(fn)
    # read time steps (convert to seconds)
    times = ds['time'][:]*60*60
    nTargetTimes = timepoints.size-1
    
    data = torch.zeros(ininames_in.size)
    target = torch.zeros(nTargetTimes*spcnames_out.size)
    met = torch.zeros((met_names.size,timepoints.size))
    # read data
    for i, name in enumerate(ininames_in):
        data_raw = torch.tensor(ds[name][:])
        
        #data[i] = fixTimesteps(data_raw, times, np.array([timepoints[0]]))
        data[i] = fixTimesteps(data_raw, times, np.array([timepoints[0]]))

    for i, name in enumerate(spcnames_out):
        data_raw = torch.tensor(ds[name][:])
        
        target[i*nTargetTimes:(i+1)*nTargetTimes] = fixTimesteps(data_raw, times, timepoints[1:])

    for i, name in enumerate(met_names):
        data_raw = torch.tensor(ds[name][:])
        met[i,:] = fixTimesteps(data_raw, times, timepoints)

    #print(data, max(abs(target)))
    ds.close()
    return data, target, met


def load_packed_data(BSP, nsamples, spc_names, emis_names, met_names, timepoints, val_perc, test_perc):

    data_tr, target_tr, met_tr, data_val, target_val, met_val, data_te, target_te, met_te =\
        load_data(BSP, nsamples, np.array([]), emis_names, spc_names, met_names, np.array([0.0, *timepoints]), val_perc, test_perc)

    target = [target_tr, target_val, target_te]
    met_pre = [met_tr, met_val, met_te]

    conc = {}
    conc["train"] = np.zeros((target_tr.shape[0], timepoints.size, spc_names.size))
    conc["val"]   = np.zeros((target_val.shape[0], timepoints.size, spc_names.size))
    conc["test"]  = np.zeros((target_te.shape[0], timepoints.size, spc_names.size))
    
    met = {}
    met["train"] = np.zeros((target_tr.shape[0], timepoints.size, met_names.size))
    met["val"]   = np.zeros((target_val.shape[0], timepoints.size, met_names.size))
    met["test"]  = np.zeros((target_te.shape[0], timepoints.size, met_names.size))

    emis = {}
    emis["train"] = data_tr
    emis["val"]   = data_val
    emis["test"]  = data_te


    nTimes = timepoints.size
    nspc = spc_names.size
    for iSc, scenario in enumerate(conc.keys()):
        for iSample in range(conc[scenario].shape[0]):
            for iT in range(nTimes):
                conc[scenario][iSample, iT, :] = target[iSc][iSample, [iT+nTimes*i for i in range(nspc)]]

    # TODO: avoid transposing, create it correctly
    for iSc, scenario in enumerate(met.keys()):
        for iSample in range(met[scenario].shape[0]):
            met[scenario][iSample,:,:] = np.transpose(met_pre[iSc][iSample,:,1:])

    return conc, met, emis


def load_data(BSP, nsamples, ininames_in, emisnames_in, spcnames_out, met_names, timepoints, val_perc, test_perc):
    
    if len(str(spcnames_out).replace("'","").replace(" ","-"))>100:
        path_dataset =  "StoredData/data_"+BSP+\
                    "_"+str(nsamples)+\
                    "_inINI"+str(ininames_in).replace("'","").replace(" ","-")+\
                    "_inEMIS"+str(emisnames_in).replace("'","").replace(" ","-")+\
                    "_out["+str(spcnames_out.size)+"_spc]"+\
                    "_met"+str(met_names).replace("'","").replace(" ","-")+\
                    "_time"+"{:.2f}".format(timepoints[0])+"_"+"{:.2f}".format(timepoints[-1])+"_n"+str(timepoints.size)+\
                    "_val"+str(int(val_perc*100))+\
                    "_test"+str(int(test_perc*100))+\
                    ".npy"
    else:
        path_dataset =  "StoredData/data_"+BSP+\
                    "_"+str(nsamples)+\
                    "_inINI"+str(ininames_in).replace("'","").replace(" ","-")+\
                    "_inEMIS"+str(emisnames_in).replace("'","").replace(" ","-")+\
                    "_out"+str(spcnames_out).replace("'","").replace(" ","-")+\
                    "_met"+str(met_names).replace("'","").replace(" ","-")+\
                    "_time"+"{:.2f}".format(timepoints[0])+"_"+"{:.2f}".format(timepoints[-1])+"_n"+str(timepoints.size)+\
                    "_val"+str(int(val_perc*100))+\
                    "_test"+str(int(test_perc*100))+\
                    ".npy"

    
    timer_arb = time.perf_counter()        
    
    # either load existing .npy data
    if os.path.isfile(path_dataset):
    
        print("  Reading existing data set. "+path_dataset)

        [data_tr, target_tr, met_tr, data_val, target_val, met_val, data_te, target_te, met_te] = np.load(path_dataset, allow_pickle=True)
    
    # or read new set from netcdf and save
    else:
        print("  Creating data set from netcdf. "+path_dataset)
        data_tr, target_tr, met_tr, data_val, target_val, met_val, data_te, target_te, met_te = \
                datasetNetCDF(BSP, nsamples, ininames_in, emisnames_in, spcnames_out, met_names, timepoints, val_perc, test_perc)
        
        np.save(path_dataset, np.array([data_tr, target_tr, met_tr, data_val, target_val, met_val, data_te, target_te, met_te], dtype=object), allow_pickle=True)

    time_IO = time.perf_counter() - timer_arb
    print("  ... Done. Time reading data: ",convertTime(time_IO),5*'         ',"\n")

    return data_tr, target_tr, met_tr, data_val, target_val, met_val, data_te, target_te, met_te

def datasetNetCDF(BSP, nsamples, ininames_in, emisnames_in, spcnames_out, met_names, timepoints, val_perc, test_perc):
    timer_arb = time.perf_counter()

    # number of samples to use for training, validation, testing
    nvalsamples = int(val_perc * nsamples)
    ntestsamples = int(test_perc * nsamples)
    ntrainsamples = nsamples - nvalsamples - ntestsamples
    nini_in = ininames_in.size
    nemis_in = emisnames_in.size
    nspc_in = nini_in + nemis_in
    nspc_out = spcnames_out.size
    nPosTimes = timepoints.size-1

    samples = np.random.permutation(nsamples)
    valsamples = samples[:nvalsamples]
    testsamples = samples[nvalsamples:nvalsamples+ntestsamples]
    trainsamples = samples[nvalsamples+ntestsamples:]

    data_tr = np.empty((ntrainsamples,nspc_in))
    target_tr = np.empty((ntrainsamples,nspc_out*nPosTimes))
    met_tr = np.empty((ntrainsamples, met_names.size, timepoints.size))

    data_val = np.empty((nvalsamples,nspc_in))
    target_val = np.empty((nvalsamples,nspc_out*nPosTimes))
    met_val = np.empty((nvalsamples, met_names.size, timepoints.size))

    data_te = np.empty((ntestsamples,nspc_in))
    target_te = np.empty((ntestsamples,nspc_out*nPosTimes))
    met_te = np.empty((ntestsamples, met_names.size, timepoints.size))


    meta_dict = np.load("../../AtCSol/NetCDF/MLData/"+BSP+"/"+BSP+"_meta.npy", allow_pickle=True)[()]
    
    # the following should be done with the ALL_specs entry of meta_dict
    sections_order = np.array(list(meta_dict["SPC_dataranges"]["GAS"].keys()))
    emis_secID = np.where(sections_order == "EMISS")[0][0]
    emis_recorded = np.array(list(meta_dict["SPC_dataranges"]["GAS"]["EMISS"].keys()))
    emisID_in = np.array([np.where(emis_recorded == i)[0][0] for i in emisnames_in])

    for isample, sample in enumerate(trainsamples,0):
        print('    Loading sample ',isample,' of ',nsamples,'. Time elapsed: ',convertTime(time.perf_counter()-timer_arb),\
                ' / est. ', convertTime(nsamples/(isample+1) * (time.perf_counter()-timer_arb)),5*'         ', end='\r')
        data, target, met = DataFromNetCDF(BSP, sample, ininames_in, spcnames_out, met_names, timepoints)
        data_tr[isample,:nini_in] = data
        if nemis_in>0:
            data_tr[isample,nini_in:] = meta_dict["ALL_tuples_noised"][sample][emis_secID][emisID_in]
        target_tr[isample,:] = target
        met_tr[isample,:,:] = met
    
    for isample, sample in enumerate(valsamples,0):
        print('    Loading sample ',isample+ntrainsamples,' of ',nsamples,'. Time elapsed: ',convertTime(time.perf_counter()-timer_arb),\
                ' / est. ', convertTime(nsamples/(isample+ntrainsamples) * (time.perf_counter()-timer_arb)),5*'         ', end='\r')
        data, target, met = DataFromNetCDF(BSP, sample, ininames_in, spcnames_out, met_names, timepoints)
        data_val[isample,:nini_in] = data
        if nemis_in>0:
            data_val[isample,nini_in:] = meta_dict["ALL_tuples_noised"][sample][emis_secID][emisID_in]
        target_val[isample,:] = target
        met_val[isample,:,:] = met
    
    for isample, sample in enumerate(testsamples,0):
        print('    Loading sample ',isample+ntrainsamples+nvalsamples,' of ',nsamples,'. Time elapsed: ',convertTime(time.perf_counter()-timer_arb),\
                ' / est. ', convertTime(nsamples/(isample+ntrainsamples+nvalsamples) * (time.perf_counter()-timer_arb)),5*'         ', end='\r')
        data, target, met = DataFromNetCDF(BSP, sample, ininames_in, spcnames_out, met_names, timepoints)
        data_te[isample,:nini_in] = data
        if nemis_in>0:
            data_te[isample,nini_in:] = meta_dict["ALL_tuples_noised"][sample][emis_secID][emisID_in]
        target_te[isample,:] = target
        met_te[isample,:,:] = met
    
    return data_tr, target_tr, met_tr, data_val, target_val, met_val, data_te, target_te, met_te

def firstgreaterentry(l, x):
    """AUX: return index of the first entry greater than x in l"""
    for i in range(l.size):
        if l[i]>x: return i

def convertTime(time):
    if time<0:
        return "negative"
    
    m = time / 60
    h = m / 60
    d = h / 24
    m=int(m)
    h=int(h)
    d=int(d)
    time=int(time)
    if time<0:
        return "0s"
    if d>0:
        timeformat = str(d)+"d "+str(h-d*24)+"h"
    elif h>0:
        timeformat = str(h)+"h "+str(m-h*60)+"m"
    elif m>0:
        timeformat = str(m)+"m "+str(time-m*60)+"s"
    else:
        timeformat = str(time)+"s"

    return timeformat


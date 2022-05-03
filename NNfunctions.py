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


def DataFromNetCDF(BSP, iFile, spcnames, metnames, timepoints):
    fn = "../../AtCSol/NetCDF/MLData/"+BSP+"/"+BSP+"_"+str(iFile)+".nc"
    ds=nc.Dataset(fn)
    
    # read time steps (convert to seconds)
    times = ds['time'][:]*60*60

    # check for availability of timepoints
    if timepoints[0]+1E-12<times[0]:
        print("First available time in file "+fn+": "+str(times[0])+". Requested: "+str(timepoints[0])+" - Abort!")
        sys.exit()
    if timepoints[-1]-1E-12>times[-1]:
        print("Last available time in file "+fn+": "+str(times[-1])+". Requested: "+str(timepoints[-1])+" - Abort!")
        sys.exit()

    conc_filedata = torch.zeros((timepoints.size, spcnames.size))
    met_filedata  = torch.zeros((timepoints.size, metnames.size))
    
    # read data
    for iSpc, spc in enumerate(spcnames):
        data_raw = torch.tensor(ds[spc][:])
        conc_filedata[:,iSpc] = fixTimesteps(data_raw, times, timepoints)

    for iMet, met in enumerate(metnames):
        data_raw = torch.tensor(ds[met][:])
        met_filedata[:,iMet]  = fixTimesteps(data_raw, times, timepoints)

    ds.close()
    return conc_filedata, met_filedata


def read_data(BSP, nFiles, spcnames, metnames, emisnames, timepoints, val_perc, test_perc):
    timer_arb = time.perf_counter()

    # number of files to use for training, validation, testing
    nvalfiles   = int(val_perc * nFiles)
    ntestfiles  = int(test_perc * nFiles)
    ntrainfiles = nFiles - nvalfiles - ntestfiles
    nEmis = emisnames.size
    nSpc  = spcnames.size
    nMet  = metnames.size

    sample_files = np.random.permutation(nFiles)
    valFiles     = sample_files[:nvalfiles]
    testFiles    = sample_files[nvalfiles:nvalfiles+ntestfiles]
    trainFiles   = sample_files[nvalfiles+ntestfiles:]

    conc = {}
    conc["train"] = np.zeros((ntrainfiles, timepoints.size, spcnames.size))
    conc["val"]   = np.zeros((nvalfiles,   timepoints.size, spcnames.size))
    conc["test"]  = np.zeros((ntestfiles,  timepoints.size, spcnames.size))
    
    met = {}
    met["train"]  = np.zeros((ntrainfiles, timepoints.size, metnames.size))
    met["val"]    = np.zeros((nvalfiles,   timepoints.size, metnames.size))
    met["test"]   = np.zeros((ntestfiles,  timepoints.size, metnames.size))

    emis = {}
    emis["train"] = np.zeros((ntrainfiles, emisnames.size))
    emis["val"]   = np.zeros((nvalfiles,   emisnames.size))
    emis["test"]  = np.zeros((ntestfiles,  emisnames.size))

    meta_dict = np.load("../../AtCSol/NetCDF/MLData/"+BSP+"/"+BSP+"_meta.npy", allow_pickle=True)[()]
    
    # the following should be done with the ALL_specs entry of meta_dict
    sections_order = np.array(list(meta_dict["SPC_dataranges"]["GAS"].keys()))
    emis_secID = np.where(sections_order == "EMISS")[0][0]
    emis_recorded = np.array(list(meta_dict["SPC_dataranges"]["GAS"]["EMISS"].keys()))
    emisIDs = np.array([np.where(emis_recorded == i)[0][0] for i in emisnames])


    categories = ["train", "val", "test"]
    catFiles = [trainFiles, valFiles, testFiles]
    for iCat, cat in enumerate(categories):
        for iFile, file in enumerate(catFiles[iCat]):
            print('    Loading NetCDF-file ',iFile,' of '+str(nFiles)+'. Time elapsed: ',convertTime(time.perf_counter()-timer_arb),\
                  ' / est. ', convertTime(nFiles/(iFile+1) * (time.perf_counter()-timer_arb)),5*'         ', end='\r')
            conc_filedata, met_filedata = DataFromNetCDF(BSP, file, spcnames, metnames, timepoints)
            if nEmis>0:
                emis[cat][iFile,:] = meta_dict["ALL_tuples_noised"][file][emis_secID][emisIDs]
            conc[cat][iFile,:,:] = conc_filedata
            met[cat][iFile,:,:]  = met_filedata

    return conc, met, emis


def get_data(BSP, nFiles, spcnames, metnames, emisnames, timepoints, val_perc, test_perc):
    
    if len(str(spcnames).replace("'","").replace(" ","-"))>100:
        path_dataset =  "StoredData/data_"+BSP+\
                    "_"+str(nFiles)+\
                    "_spc["+str(spcnames.size)+"_spc]"+\
                    "_met"+str(metnames).replace("'","").replace(" ","-")+\
                    "_emis"+str(emisnames).replace("'","").replace(" ","-")+\
                    "_time"+"{:.2f}".format(timepoints[0])+"_"+"{:.2f}".format(timepoints[-1])+"_n"+str(timepoints.size)+\
                    "_val"+str(int(val_perc*100))+\
                    "_test"+str(int(test_perc*100))+\
                    ".npy"
    else:
        path_dataset =  "StoredData/data_"+BSP+\
                    "_"+str(nFiles)+\
                    "_spc"+str(spcnames).replace("'","").replace(" ","-")+\
                    "_met"+str(metnames).replace("'","").replace(" ","-")+\
                    "_emis"+str(emisnames).replace("'","").replace(" ","-")+\
                    "_time"+"{:.2f}".format(timepoints[0])+"_"+"{:.2f}".format(timepoints[-1])+"_n"+str(timepoints.size)+\
                    "_val"+str(int(val_perc*100))+\
                    "_test"+str(int(test_perc*100))+\
                    ".npy"

    
    timer_arb = time.perf_counter()        
    
    # either load existing .npy data
    if os.path.isfile(path_dataset):
    
        print("  Reading existing data set. "+path_dataset)

        [conc, met, emis] = np.load(path_dataset, allow_pickle=True)
    
    # or read new set from netcdf and save
    else:
        print("  Creating data set from netcdf. "+path_dataset)
        conc, met, emis = read_data(BSP, nFiles, spcnames, metnames, emisnames, timepoints, val_perc, test_perc)

        np.save(path_dataset, np.array([conc, met, emis], dtype=object), allow_pickle=True)


    time_IO = time.perf_counter() - timer_arb
    print("  ... Done. Time reading data: ",convertTime(time_IO),5*'         ',"\n")

    return conc, met, emis


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


import sys
import numpy as np

def EmisSplit(GroupedEmis, GivenGroupNames=[""]):

   # INTEGER :: nGroups, i, ii, j, iName
   # CHARACTER(LenName), ALLOCATABLE :: GroupNames(:), GroupNamesAll(:)

    # NOT STANDALONE : LenName, dp

    class EmisGroup:
        def __init__(self, name, members, portions):
            self.portions = portions
            self.members = members
            self.name = name

        @classmethod
        def empty(cls):
            return cls('None',['None'],[float('nan')])

   
    # alphabetically ordered names of all (implemented) groups 
    GroupNamesAll = np.array([ 'ALD','API','CSL','HC3','HC5','HC8','KET','OLI','OLT','TOL','XYL' ])

    nGroups = len(GroupedEmis)

    # check for errors with name lists
    if (nGroups != len(GroupNamesAll)):
        if (not ((GivenGroupNames!=[""]) and len(GivenGroupNames)==nGroups)):
            print('SplitEmis :: Unable to assign emission groups. Provide specific group names or fill missing groups with 0.')
            sys.exit()

    if (GivenGroupNames!=""):
        # use given groups
        GroupNames = np.array(GivenGroupNames)
    else:
        # use all
        GroupNames = np.array(GroupNamesAll)

    # define all groups 
    # NOTE: same order as in GroupNamesAll needed !
    # ------------------------------------------------------------------------------------
    Groups = [EmisGroup.empty() for i in range(len(GroupNamesAll))]
    Groups[0].name      = 'ALD'
    Groups[0].portions  = [  0.595588235294118           \
                           , 0.132352941176471           \
                           , 0.0882352941176471          \
                           , 0.0441176470588235          \
                           , 0.0367647058823529          \
                           , 0.0294117647058824          \
                           , 0.0294117647058824          \
                           , 0.0294117647058824          \
                           , 0.0147058823529412          ]
    Groups[0].members   = [  'CC=O'                      \
                           , 'O=CC=C'                    \
                           , 'c1ccccc1C=O'               \
                           , 'Cc1ccc(C=O)cc1'            \
                           , 'O=CC=O'                    \
                           , 'CC(=O)C=O'                 \
                           , 'CCC=O'                     \
                           , 'CCCC=O'                    \
                           , 'CC=CC=O'                   ]

    Groups[1].name      = 'API'
    Groups[1].portions  = [  0.6                         \
                           , 0.4                         ] 
    Groups[1].members   = [  'C12CC(C1(C)C)CC=C2C'       \
                           , 'CC1(C)C(CC12)CCC2=C'       ] 

    Groups[2].name      = 'CSL'
    Groups[2].portions  = [  1.0                         ]
    Groups[2].members   = [  'c1ccc(O)cc1'               ]

    Groups[3].name      = 'HC3'
    Groups[3].portions  = [  0.0447054                   \
                           , 0.1464487                   \
                           , 0.6843783                   \
                           , 0.0185753                   \
                           , 0.0031274                   \
                           , 0.0011783                   \
                           , 0.0003189                   \
                           , 0.0089214                   \
                           , 0.0039359                   \
                           , 0.0013120                   \
                           , 0.0439346                   \
                           , 0.0432                      ] 
    Groups[3].members   = [  'CCC'                       \
                           , 'C#C'                       \
                           , 'CCCC'                      \
                           , 'CC(C)C'                    \
                           , 'CLC(CL)=C(CL)CL'           \
                           , 'CLC(CL)=CCL'               \
                           , 'CLCCCL'                    \
                           , 'CCOC(=O)C'                 \
                           , 'CC(=O)OC'                  \
                           , 'C1OC1'                     \
                           , 'CC(=O)OC(C)C'              \
                           , 'COC'                       ]
 
    Groups[4].name      = 'HC5'
    Groups[4].portions  = [  0.240478019                 \
                           , 0.137359133                 \
                           , 0.135766563                 \
                           , 0.064499071                 \
                           , 0.040212384                 \
                           , 0.02468483                  \
                           , 0.234226829                 \
                           , 0.078365854                 \
                           , 0.021768293                 \
                           , 0.016979268                 \
                           , 0.004789024                 \
                           , 0.000870732                 ]
    Groups[4].members   = [  'CCC(C)C'                   \
                           , 'CCCCCC'                    \
                           , 'CCCCC'                     \
                           , 'CCCC(C)C'                  \
                           , 'CCC(C)CC'                  \
                           , 'CC(C)C(C)C'                \
                           , 'CC(O)C'                    \
                           , 'CCCCOC(=O)C'               \
                           , 'CCCOC(=O)C'                \
                           , 'CLC=C'                     \
                           , 'CCCCO'                     \
                           , 'CCCO'                      ]

    Groups[5].name      = 'HC8'
    Groups[5].portions  = [  0.408784068                 \
                           , 0.098610192                 \
                           , 0.087852716                 \
                           , 0.076198785                 \
                           , 0.073509416                 \
                           , 0.063648397                 \
                           , 0.036754708                 \
                           , 0.026893689                 \
                           , 0.078025743                 \
                           , 0.042837663                 \
                           , 0.005354708                 \
                           , 0.000764958                 \
                           , 0.000764958                 ]
    Groups[5].members   = [  'CCCCCCC'                   \
                           , 'CCCC(C)C'                  \
                           , 'CCCCCCCC'                  \
                           , 'CCCC(C)CC'                 \
                           , 'CCCCCCCCCCC'               \
                           , 'C1CCCCC1'                  \
                           , 'CCCCCCCCC'                 \
                           , 'CCCCCCCCCC'                \
                           , 'COCCO'                     \
                           , 'CC(O)CO'                   \
                           , 'CCOCC'                     \
                           , 'CCC(O)C'                   \
                           , 'C1CCC(O)CC1'               ]

    Groups[6].name      = 'KET'
    Groups[6].portions  = [  0.516                       \
                           , 0.00436036                  \
                           , 0.00436036                  \
                           , 0.475279279                 ]
    Groups[6].members   = [  'CC(=O)C'                   \
                           , 'C1CCC(=O)CC1'              \
                           , 'CCCCC(=O)C'                \
                           , 'CCC(=O)C'                  ]

    Groups[7].name      = 'OLI'
    Groups[7].portions  = [  0.331507491                 \
                           , 0.157229267                 \
                           , 0.117448368                 \
                           , 0.098505083                 \
                           , 0.08903344                  \
                           , 0.071984484                 \
                           , 0.035992242                 \
                           , 0.017048957                 \
                           , 0.081250668                 ]
    Groups[7].members   = [  'C=CC=C'                    \
                           , 'CC=C(C)C'                  \
                           , 'CC\C=C\C'                  \
                           , 'CC=CC'                     \
                           , 'cCC=CC'                    \
                           , 'CC/C=C\C'                  \
                           , 'CCC/C=C\C'                 \
                           , 'CC1(C)C(CC12)CCC2=C'       \
                           , 'CCCCC=C'                   ]


    Groups[8].name      = 'OLT'
    Groups[8].portions  = [  0.626                       \
                           , 0.133510949                 \
                           , 0.089671533                 \
                           , 0.029890511                 \
                           , 0.015941606                 \
                           , 0.002989051                 \
                           , 0.00099635                  \
                           , 0.055                       \
                           , 0.045                       ]
    Groups[8].members   = [  'CC=C'                      \
                           , 'CCC=C'                     \
                           , 'CCCC=C'                    \
                           , 'CCCCC=C'                   \
                           , 'CC(C)C=C'                  \
                           , 'CC(C)=C'                   \
                           , 'CCC(C)=C'                  \
                           , 'CCCCC=C'                   \
                           , 'c1ccccc1C=C'               ]

    Groups[9].name     = 'TOL' 
    Groups[9].portions = [   0.111                       \
                           , 0.04                        \
                           , 0.836505957                 \
                           , 0.009122635                 \
                           , 0.002644242                 \
                           , 0.000727167                 ]
    Groups[9].members  = [   'c1ccccc1'                  \
                           , 'c1ccccc1C=C'               \
                           , 'c1ccccc1C'                 \
                           , 'c1ccccc1CC'                \
                           , 'c1ccccc1CCC'               \
                           , 'c1ccccc1C(C)C'             ]

    Groups[10].name     = 'XYL'
    Groups[10].portions = [  0.206094627                 \
                           , 0.277329798                 \
                           , 0.176655749                 \
                           , 0.1629968                   \
                           , 0.075517034                 \
                           , 0.050082816                 \
                           , 0.03127506                  \
                           , 0.020048115                 ]
    Groups[10].members  = [  'Cc1ccccc1C'                \
                           , 'c1cc(C)ccc1C'              \
                           , 'Cc1ccc(C)cc1C'             \
                           , 'c1c(C)cc(C)cc1C'           \
                           , 'c1c(C)cccc1C'              \
                           , 'Cc1cccc(C)c1C'             \
                           , 'Cc1ccccc1CC'               \
                           , 'c1c(C)cccc1CC'             ]


    # control groups
    #for i in range(len(Groups)):
    #    if (abs(sum(Groups[i].portions)-1)>1e-16):
    #        print('SplitEmis :: Portions of group ', Groups[i].name, ' do not sum up to 1. (', sum(Groups[i].portions), ')')
    # ------------------------------------------------------------------------------------

    ii=0
    for i in range(nGroups):
        iName = np.where(GroupNamesAll == GroupNames[i])[0][0]
        ii = ii + len(Groups[iName].members)

    Emis = [0.0 for i in range(ii)]
    EmisNames = ["" for i in range(ii)]

    ii=0
    for i in range(nGroups):
        iName = np.where(GroupNamesAll == GroupNames[i])[0][0]
        for j in range(len(Groups[iName].members)):
            EmisNames[ii] = Groups[iName].members[j]
            Emis[ii]      = Groups[iName].portions[j]*GroupedEmis[i]
            ii=ii+1

    return Emis, EmisNames

# TEST
#DO i=1,SIZE(Emis)
#  WRITE(*,*) TRIM(EmisNames(i)), Emis(i)
#END DO

## Individual-level Functional Connectivity Predicts Cognitive Control Efficiency

Repository of code used to run our SVR model in the paper by Deck 2023: Individual-level Functional Connectivity Predicts Cognitive Control Efficiency

### Step-by-step guide to reproduce results

In order to reproduce our results we provide guidelines on how to interact with our code and corresponding [data set](https://osf.io/d7p58/)

#### Main SVR analysis
1. Users should download the data at our OSF [database](https://osf.io/d7p58/)

2. Amend the paths to ```wrapper_synth_svr.py``` to where the OSF data was downloaded including functional connectivity data and behavior data.
    a. Connectivity data will have a naming structure such as ```FPCNA_avg_output.csv```, while behavioral data will have the following naming structure, ```shiftcost_ACC.csv```, depending on the behavior of interest. 
    b. The connectivity data's name is based on whether or not fronto-parietal control network-A or B is incluced in the analysis and whether the lateral default-mode network is included or not.
    c. Depending on your compute capabilities the SVR pipeline can take 1-3 hours. The original paper used 8 CPU-cores and 16 gb of ram on a SLURM-Cluster through Google Cloud Platform. 

3. The results can be found in the reports folder which the user can specify the path to.

4. The replication and control data analysis follows this same process.

#### Time series reliabiltiy

1. Time series reliabiltiy data should be downloaded and unzipped from the OSF database.
2. Users should then amend the ```group_ts_reliability.py``` wrapper script for the output and other paths in this script to be pointed to the downloaded data.


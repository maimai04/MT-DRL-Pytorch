# Folder: data

## Overview
In this folder, the actual data used is saved. The folder structure is as follows:

```
raw             : here, the raw dtaa (as it was downloaded) is stored

intermediate    : here, the intermediate data (data after some preprocessing, trying things out etc.) is stored.
    
preprocessed    : here, the finalized, preprocessed data is stored, which is then actually used for the modeling.
                  It is the data that is then loaded from run.py when the run is started.
```
## scripts

This folder contains scripts to train nnunet models in a reproducible way. These scripts are based on https://github.com/neuropoly/totalspineseg/tree/4502d41bcb4a12e44f4be666411461ca81b02d89/scripts

* `download_datasets.sh`: Downloads BIDS-like and git-annexed datasets from the Neuropoly server.
* `prepare_datasets.sh`: Prepares datasets in the format required by nnUNet.
* `train.sh`: Trains nnUNet models by specifying the trainer, dataset, and GPU to use. Information about the training configuration is saved in the output nnUNet folder.

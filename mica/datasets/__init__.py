from . import data_module


DATA_MODULES = {
    
    "pretrain": data_module.PretrainingDataModuleDerm,

    # NOTE: replace dataset_name with your dataset name
    "dataset_name": data_module.DermDataModule,  # first stage
    "dataset_name_cbm": data_module.DermCBMDataModule,

}

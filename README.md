<!--
 * @Author: Peng Bo
 * @Date: 2023-02-02 08:25:04
 * @LastEditTime: 2023-02-15 18:21:04
 * @Description: 
 * 
-->
## visible-infrared fusion and image dehazing by deep learning

### fusion
#### prepare the dataset
1. download MSRS, and extract
2. process the data into h5py format, `python utils/dataprocessing.py`

#### training
1. training: `python train.py --config-file your_config_path`
  default `your_config_path` is set as `experiments/template/config.yaml`, 
`experiments/restormer_coa_fusion/config.yaml` shows how to configure it.

#### testing
1. test demo: `python train.py --config-file your_config_path --eval`: 

### image dehazing
1. test demo: `python models/aod_net.py`
2. training: coming soon.

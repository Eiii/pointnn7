# Requirements

Off the top of my head:
* torch
* Custom kaolin library (see other repo)

# Get demo datasets
Run `utils/download-modelnet10.sh` from the project's root directory to download and 
unpack ModelNet10.

A demo SC2 dataset is included at `data/sc2scene.tar.zst`. Untar it in-place to make it
available for use.

# Train networks
The `experiments` folder contains two configuration files that quickly train demo 
networks. These won't perform well, they're just meant to show off the whole pipeline.
(Increase the number of epochs / batch size in the config for better results)

To train a SC2 unit transition predictor:
`python -m pointnn ./experiments/sc2demo.json`

To train a point cloud autoencoder:
`python -m pointnn ./experiments/pcdemo.json`

# Model Evaluation Scripts

Scripts to poke at the trained models are in `pointnn.eval`. A lot of them are vestigial
at this point, so don't expect them to all work or make sense or have a consistent 
interface.

These might be helpful though:

To display Starcraft transition predictions for an entire scene:
`python -m pointnn.eval.sc2.frame --net output/sc2demo --data data/sc2scene/test`

To test a point cloud autoencoder:
`python -m pointnn.eval.model --net output/pcdemo --data data/ModelNet10`

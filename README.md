## Function Informed Clustering

:construction: Repo under construction! :construction:

More examples and more documentation coming soon! Planned updates can be found under Issues. I am currently working on uploading pre-trained models for each figure. If you are looking for something and cannot find it, please email me at schmittms@uchicago.edu. 

## Summary

This repository contains code accompanying the paper [Learning functional groups in complex microbiomes](https://www.biorxiv.org/content/10.64898/2026.03.03.709366v1).

To get started, I recommend taking a look at the demo notebook [demo_method_comparison_simulated_data.ipynb](demo_method_comparison_simulated_data.ipynb) which shows how to train SCiFI on a simple simulated dataset.

The code for the SCiFI model itself is in [utils/models.py](utils/models.py): the model with no gating is called `FC_Gumbelpredictor`, and the gated model is called `GatedModel`. The code to generate the figures in the paper are in the corresponding directories. They all have the same structure (with the exception of fig5_consumer_resource):
```
figX/
│   dataset.py
│   generate_main_figures.ipynb
│   train.py
└───trained_models/
│   └───hyperparameter_set_0/
│   │   │   model_0.pt
│   │   │   model_1.pt
│   │   │   ...
│   │   │   model_N.pt
│   └───hyperparameter_set_1/
│   │   │   ...
│   │  ...
│  ...
```

The relevant dataset class is defined in `dataset.py`, and the models are trained with `train.py`. The trained models are saved in `trained_models`, where I organize the models by the set of hyperparameters they were trained with (e.g. learning rate, etc.). I always train an ensemble of `N` models, which are saved individually. In `generate_main_figures.ipynb` I load the trained models and generate the relevant plots.

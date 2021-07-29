

import configparser
import gc
import logging
import pathlib as path
import sys
from collections import defaultdict
from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from idao.data_module import IDAODataModule
from idao.model import SimpleConv

def compute_predictions(mode, dataloader, checkpoint_path, cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info("Loading checkpoint")
    model = SimpleConv.load_from_checkpoint(checkpoint_path, mode=mode)
    model = model.cpu().eval()

    dict_pred = defaultdict(list)
    if mode == "classification":
        logging.info("Classification model loaded")
    else:
        logging.info("Regression model loaded")

    for img, name in iter(dataloader):
        if mode == "classification":
            dict_pred["id"].extend(map(lambda x: x.strip('.png'), name))
            output = model(img)["class"].detach()[:, 1].numpy()
            dict_pred["particle"].extend(output)
        else:
            output = model(img)["energy"].detach().squeeze(1).numpy()
            dict_pred["energy"].extend(output)
            
    return dict_pred


def main():
    config = configparser.ConfigParser()
    config.read("./config.ini")
    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=512, cfg=config
    )

    dataset_dm.prepare_data()
    print(dataset_dm.dataset.class_to_idx)
    dataset_dm.setup()
    dl = dataset_dm.test_dataloader()

    dict_pred = defaultdict(list)
    for mode in ["regression", "classification"]:
        if mode == "classification":
            model_path = config["REPORT"]["ClassificationCheckpoint"]
        else:
            model_path = config["REPORT"]["RegressionCheckpoint"]

        dict_pred.update(compute_predictions(mode, dl, model_path, cfg=config))

    data_frame = pd.DataFrame(dict_pred,
                              columns=["id", "energy", "particle"])
    data_frame.set_index("id", inplace=True)
    data_frame.to_csv('submission_classification.csv.gz',
                      index=True, header=True, index_label="id", columns=["particle"])
    data_frame.to_csv('submission_regression.csv.gz',
                      index=True, header=True, index_label="id", columns=["energy"])


if __name__ == "__main__":
    main()

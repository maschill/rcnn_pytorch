from itertools import product
import json
from rcnn.training import training


if __name__ == "__main__":
    with open("hparams.json") as json_file:
        hparams = json.load(json_file)
        for hps in product(*hparams.values()):
            training({hpk: hpv for hpk, hpv in zip(list(hparams.keys()), hps)})

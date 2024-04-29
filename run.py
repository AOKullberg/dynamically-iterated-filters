import logging
import pdb

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import numpy as np

import tqdm

from ssmjax.utility.pytrees import tree_stack

log = logging.getLogger(__name__)

def eval_data(alg, observations, initial_state, initial_theta):
    res = dict(ell=[], result=[])
    for sim in tqdm.tqdm(observations):
        result = alg(initial_state=initial_state, initial_theta=initial_theta,
                    observations=sim)
        res['ell'].append(result[0])
        res['result'].append(result[1])
        if len(result) > 2:
            if list(result[2].keys()) in list(res.keys()):
                for key in result[2].keys():
                    res[key].append(result[2][key])
            else:
                for key in result[2].keys():
                    res[key] = [result[2][key]]
    res['result'] = tree_stack(res['result'])
    return res

@hydra.main(version_base=None, config_path="config", config_name="se-config")
def main(cfg: DictConfig) -> None:
    log.info("Instantiating objects")
    initial_state = instantiate(cfg.sim.initial_state)
    initial_theta = instantiate(cfg.sim.initial_theta)
    data_generator = instantiate(cfg.sim.data_generator)
    alg = instantiate(cfg.alg)
    log.info("Generating data")
    x, y, _ = data_generator()
    log.info("Data generated!")
    log.info("Running data evaluation on {} simulations".format(y.shape[0]))
    res = eval_data(alg, y, initial_state, initial_theta)
    log.info("Evaluation complete")
    log.info("Saving data and quitting")
    np.savez('result.npz',
            state_mean=np.array(res['result'].mean),
            state_cov=np.array(res['result'].cov),
            ell=np.array(res['ell']),
            x=x,
            y=y)

if __name__ == "__main__":
    main()

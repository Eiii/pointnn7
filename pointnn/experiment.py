from . import nets

import json
from pathlib import Path

def read_experiment_json(path):
    """Translate an experiment JSON specification to an executable list of
    arguments for `train` to consume. """
    with open(path, 'rb') as fd:
        desc = json.load(fd)
    run_args = []
    prob_args = load_dict(path, desc.get('problem_args', dict()))
    for entry in desc['entries']:
        net_args = load_dict(path, entry.get('net_args', dict()))
        train_args = load_dict(path, entry.get('train_args', dict()))
        print(f"Experiment {entry['name']}")
        print(f'Train args: {train_args}')
        print(f'Net args: {net_args}')
        # Create network w/ given arguments
        net = nets.make_net(net_args)
        args = {'name': entry['name'],
                'net': net,
                'problem_args': prob_args,
                'train_args': train_args,
                'epochs': train_args['epochs'],
                'out_dir': desc['output_path']}
        run_args += [args]
    return run_args


def load_dict(path, val):
    base_key = 'BASE'
    base_path = Path(path).parent

    def _ld(pth):
        with (base_path/Path(pth)).open('rb') as fd:
            return json.load(fd)

    if type(val) is str:
        return _ld(val)
    elif type(val) is dict:
        if base_key in val:
            base = _ld(val[base_key])
            del val[base_key]
            updated_keys = base.keys() & val.keys()
            if updated_keys:
                print('Overwriting default values:')
                for k in updated_keys:
                    print(f'\t{k}: {base[k]} -> {val[k]}')
            base.update(val)
            return base
        else:
            return val
    else:
        raise ValueError('Unexpected config dictionary value')

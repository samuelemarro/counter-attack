import json
import os
import pathlib

names = [
    'all_attacks',
    'all_domains',
    'all_distances',
    'all_types'
]


def convert_simple(file_path):
    try:
        f = open(file_path, 'r')
        cfg = json.load(f)

        """keys = cfg['mip']['all_domains']['all_distances']['all_types'].keys()
        cfg['mip']['params'] = dict()
        for key in keys:
            cfg['mip']['params'][key] = cfg['mip']['all_domains']['all_distances']['all_types'][key]

        del cfg['mip']['all_domains']
        print(cfg)"""
        """cfg['overrides'] = dict()
        cfg['overrides']['mip'] = cfg['mip']
        del cfg['mip']"""
        cfg['mip'] = cfg['overrides']['mip']
        del cfg['overrides']

        f = open(file_path, 'w')
        json.dump(cfg, f, indent=4)
    except:
        pass

def iterate(dd):
    for name in os.listdir(dd):
        path = pathlib.Path(dd) / name
        if str(path).endswith('.cfg'):
            convert_simple(path)
        if path.is_dir():
            iterate(str(path))

iterate('attack_configurations')

#convert_simple('mip_test_configuration.cfg')
import argparse
import itertools
import os
from typing import Any, Iterable, Iterator, Mapping, Sequence


class HParamAction(argparse.Action):

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(HParamAction, self).__init__(option_strings, dest, nargs="*", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2:
            raise argparse.ArgumentError(self, (
                "At least 2 values required. The first value is treated as the name of the "
                "hyper-parameter and subsequent values are treated as possible values."
            ))
        params = getattr(namespace, self.dest)
        if params is None:
            params = {}
            setattr(namespace, self.dest, params)

        if values[0] in params:
            raise argparse.ArgumentError(self, f"'{values[0]}' was specified more than once.")

        params[values[0]] = values[1:]


parser = argparse.ArgumentParser()

parser.add_argument("dir", help="Top level directory for the configs.")
parser.add_argument("-p", action=HParamAction)
args = parser.parse_intermixed_args()


def generate_configs(hparams: Mapping[str, Iterable]) -> Iterator[Mapping[str, Any]]:
    names = list(hparams.keys())
    all_values = list(hparams.values())
    for values in itertools.product(*all_values):
        yield dict(zip(names, values))


def dict_to_config_bindings(values: Mapping[str, Any]) -> Sequence[str]:
    return [f"{name} = {val}\n" for name, val in values.items()]


dir = os.path.expanduser(args.dir)
os.makedirs(dir)
for i, config in enumerate(generate_configs(args.p)):
    with open(os.path.join(dir, f"{i}.gin"), "x") as f:
        f.writelines(dict_to_config_bindings(config))

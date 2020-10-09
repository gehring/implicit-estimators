from absl import app
from absl import flags

import gin

from experiments import gridsearch

if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file", None, "List of paths to the config files.")
    flags.DEFINE_multi_string(
        "gin_param", None, "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # read config file(s).
    gin_configs = []
    for config_file in FLAGS.gin_file:
        with open(config_file, "r") as file:
            gin_configs.append(file.read())

    gin_params = FLAGS.gin_param or []
    gin.parse_config(gin_configs)
    gin.parse_config(gin_params)

    config = {
        "SEED": [167442891167852401332641527644475489738,
                 179293636264133832658649852078596007641,
                 298791047951749113491137367315834136104,
                 70260435749367869108637445047908730968,
                 106821836386322972690649687400811145885,
                 184497179512677843888352218018501684453,
                 266022625291476938506617656772211740905,
                 83458978794483395332020155565442813967,
                 169103010037718488704511065537885528924,
                 20010431513504658981827741904466217880,
                 196843368530108191944994326047971450656,
                 40744976603229448190480298232212623300,
                 72910582825604166502825312522689323832,
                 210108443496965127478500887752901789945,
                 206738396012001355474695064529691443996,
                 13619745772645856733194229134378625579,
                 84269579424565118101964105499073671178,
                 156859006390783818765182231993891547469,
                 317603931917959961529085008539732470979,
                 101382007775519830691764142979738864793],
        "LEARNING_RATE": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        "BATCH_SIZE": [1, 4, 10],
    }
    gridsearch.launch(config, gin_configs=gin_configs, gin_params=gin_params)


if __name__ == "__main__":
    app.run(main)

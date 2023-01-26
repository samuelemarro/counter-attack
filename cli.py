# PyTorch has some serious bugs concerning dll loading:
# If PyTorch is loaded before Julia, Julia's import fails.
# We therefore import Julia before anything else
try:
    from julia.api import JuliaInfo
    info = JuliaInfo.load()
    if not info.is_compatible_python():
        print('Julia: using non-compiled modules.')
        from julia.api import Julia
        jl = Julia(compiled_modules=False)

    import julia
    from julia import Base
except:
    # Silent failure, if the program actually
    # needs Julia it will re-raise an error
    pass

import logging

import click

import commands


@click.group()
def main():
    pass



main.add_command(commands.accuracy)
main.add_command(commands.attack)
main.add_command(commands.attack_matrix)
main.add_command(commands.compare)
main.add_command(commands.cross_validation)
main.add_command(commands.distance_dataset)
main.add_command(commands.evasion)
main.add_command(commands.mip)
main.add_command(commands.perfect_approximation)
main.add_command(commands.prune_relu)
main.add_command(commands.prune_weights)
main.add_command(commands.train_approximator)
main.add_command(commands.train_classifier)
main.add_command(commands.tune_mip)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s:%(levelname)s: %(message)s')
    main()

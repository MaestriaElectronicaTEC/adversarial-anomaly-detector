""" Tool for training the models"""

import sys
import argparse

from models.DCGAN import DCGAN

#----------------------------------------------------------------------------

def train_gan(latent_dim, datadir, results_dir):
    gan  = DCGAN(latent_dim, results_dir)
    gan.preprocessing(datadir)
    gan.train()

#----------------------------------------------------------------------------

def cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for training the models in the Adversarial Anomaly Detector.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'train_gan',            'Training of the DCGAN model.')

    p.add_argument(     '--latentDim',          help='Latent space dimension of the GAN\'s generator', type=int, default=100)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline(sys.argv)

#----------------------------------------------------------------------------

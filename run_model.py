""" Tool for use the model capabilities """

import sys
import argparse

from models.DCGAN import DCGAN

#----------------------------------------------------------------------------

def generate_samples(generatorDir, discriminatorDir, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
}

    gan  = DCGAN(100, resultsDir)
    gan.load(modelDir)
    gan.preprocessing(dataDir)
    gan.plot()

#----------------------------------------------------------------------------

def cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for use the prediction capabilities of the models in the Adversarial Anomaly Detector.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'generate_samples',     'Generate sintetic images from the GAN\'s generator.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
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

""" Tool for training the models"""

import sys
import pickle
import argparse

from models.DCGAN import DCGAN
from models.AAD import AAD

#----------------------------------------------------------------------------

def train_gan(latentDim, epochs, dataDir, resultsDir):
    gan  = DCGAN(latentDim, resultsDir)
    gan.preprocessing(dataDir)
    gan.train(n_epochs=epochs)
    # save the object state
    pickle.dump( gan, open( resultsDir + "/gan.pkl", "wb" ) )

def train_anomaly_detector(generatorDir, discriminatorDir, latentDim, epochs, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.train(n_epochs=epochs)
    # save the object state
    pickle.dump( anomaly_detector, open( resultsDir + "/add.pkl", "wb" ) )

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
    p.add_argument(     '--epochs',             help='Number of epochs for the training', type=int, default=20)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'train_anomaly_detector','Training of the Adversarial Anomaly Detector model.')

    p.add_argument(     '--generatorDir',       help='Path of the generator h5 file', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the discriminator h5 file', default='')
    p.add_argument(     '--latentDim',          help='Latent space dimension of the GAN\'s generator', type=int, default=100)
    p.add_argument(     '--epochs',             help='Number of epochs for the training', type=int, default=20)
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

""" Tool for use the model capabilities """

import sys
import argparse

from models.DCGAN import DCGAN
from models.AAD import AAD
from utils.PreProcessing import load_test_data

#----------------------------------------------------------------------------

def generate_samples(generatorDir, discriminatorDir, latentDim, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)
    gan.preprocessing(dataDir)
    gan.plot()

def evaluate_anomaly(generatorDir, discriminatorDir, anomalyDetectorDir, latentDim, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(anomalyDetectorDir)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.plot()

def analyze_anomalies(generatorDir, discriminatorDir, anomalyDetectorDir, latentDim, anomalyTreshold, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(anomalyDetectorDir)
    anomaly_detector.analize_anomalies(load_test_data(dataDir), anomalyTreshold)

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
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'evaluate_anomaly',     'Remove anomalies from some input image.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'analyze_anomalies',    'Analysis of anomalies from a dataset.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--anomalyTreshold',          help='Anomaly scroe treshold', type=int, default=2000)
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

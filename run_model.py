""" Tool for use the model capabilities """

import sys
import argparse

from models.AAD import AAD
from models.DCGAN import DCGAN
from models.StyleAAD2 import StyleAAD2
from models.stylegan import StyleGAN_G
from models.stylegan import StyleGAN_D
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

def plot_anomaly(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, dataDir, resultsDir):
    ganModelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(ganModelDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : SVCDir,
        "scaler" : ScalerDir
    }

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.plot()

def plot_style_anomaly(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, dataDir, resultsDir):
    # load the style gan
    style_gan_g = StyleGAN_G()
    style_gan_g.load_weights(generatorDir)

    style_gan_d = StyleGAN_D()
    style_gan_d.load_weights(discriminatorDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : SVCDir,
        "scaler" : ScalerDir
    }

    anomaly_detector = StyleAAD2(style_gan_g, style_gan_d, resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.plot()

def plot_anomaly_batch(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, dataDir, dataBatch, resultsDir):
    # load the style gan
    style_gan_g = StyleGAN_G()
    style_gan_g.load_weights(generatorDir)

    style_gan_d = StyleGAN_D()
    style_gan_d.load_weights(discriminatorDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : SVCDir,
        "scaler" : ScalerDir
    }

    anomaly_detector = StyleAAD2(style_gan_g, style_gan_d, resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.plot_batch(dataBatch)

def evaluate_anomaly_detector(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, normlaDataDir, anomalyDataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : SVCDir,
        "scaler" : ScalerDir
    }

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)

    normal = load_test_data(normlaDataDir)
    anomaly = load_test_data(anomalyDataDir)

    anomaly_detector.evaluate_model(normal, anomaly)

def evaluate_anomalies_in_dataset(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : SVCDir,
        "scaler" : ScalerDir
    }

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)

    dataset = load_test_data(dataDir)

    normal, anomaly = anomaly_detector.evaluate_subset(dataset)
    anomaly_detector.t_sne_analysis(normal, anomaly)

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

    p = add_command(    'plot_anomaly',         'Remove anomalies from some input image.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'plot_style_anomaly',         'Remove anomalies from some input image.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'plot_anomaly_batch',   'Remove anomalies from a batch of images.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--dataBatch',          help='Batch size', type=int, default=1)
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'evaluate_anomaly_detector', 'Plot anomalies from a dataset.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--normlaDataDir',      help='Path of the dataset', default='')
    p.add_argument(     '--anomalyDataDir',     help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'evaluate_anomalies_in_dataset', 'Evaluate the presence of anomalies in some dataset')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
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

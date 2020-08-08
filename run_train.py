""" Tool for training the models"""

import sys
import pickle
import argparse

from models.AAD import AAD
from models.StyleAAD import StyleAAD
from models.DCGAN import DCGAN
from models.stylegan import StyleGAN_G
from models.stylegan import StyleGAN_D
from utils.PreProcessing import load_test_data

#----------------------------------------------------------------------------

def train_gan(latentDim, epochs, dataDir, resultsDir):
    gan  = DCGAN(latentDim, resultsDir)
    gan.preprocessing(dataDir)
    gan.train(n_epochs=epochs)
    # save the object state
    pickle.dump( gan, open( resultsDir + "/gan.pkl", "wb" ) )

def train_anomaly_detector(generatorDir, discriminatorDir, latentDim, reconstructionError, dicriminatorError, epochs, dataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim, reconstructionError, dicriminatorError)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.train(n_epochs=epochs)
    # save the object state
    #pickle.dump( anomaly_detector, open( resultsDir + "/add.pkl", "wb" ) )

def train_style_anomaly_detector(generatorDir, discriminatorDir, latentDim, reconstructionError, dicriminatorError, epochs, dataDir, resultsDir):
    # load the style gan
    style_gan_g = StyleGAN_G()
    style_gan_g.load_weights(generatorDir)

    style_gan_d = StyleGAN_D()
    style_gan_d.load_weights(discriminatorDir)

    anomaly_detector = StyleAAD(style_gan_g, style_gan_d, resultsDir, latentDim, reconstructionError, dicriminatorError)
    anomaly_detector.preprocessing(dataDir)
    anomaly_detector.train(n_epochs=epochs)

def train_anomaly_grid_search(generatorDir, discriminatorDir, anomalyDetectorDir, latentDim, normlaDataDir, anomalyDataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : '',
        "scaler" : ''
    }

    normal = load_test_data(normlaDataDir)
    anomaly = load_test_data(anomalyDataDir)
    
    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)
    anomaly_detector.train_svm_with_grid_search(normal, anomaly)

def train_anomaly_classifier(generatorDir, discriminatorDir, anomalyDetectorDir, latentDim, C, gamma, kernel, degree, normlaDataDir, anomalyDataDir, resultsDir):
    modelDir = {
        "generator": generatorDir,
        "discriminator": discriminatorDir
    }

    gan  = DCGAN(latentDim, resultsDir)
    gan.load(modelDir)

    aadModelDir = {
        "aad" : anomalyDetectorDir,
        "svc" : '',
        "scaler" : ''
    }

    normal = load_test_data(normlaDataDir)
    anomaly = load_test_data(anomalyDataDir)
    
    anomaly_detector = AAD(gan.get_generator(), gan.get_discriminator(), resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)
    anomaly_detector.train_svm(C, gamma, degree, kernel, normal, anomaly)

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
    p.add_argument(     '--reconstructionError',help='Reconstruction error weight', type=float, default=0.90)
    p.add_argument(     '--dicriminatorError',  help='Discriminator error weight', type=float, default=0.10)
    p.add_argument(     '--epochs',             help='Number of epochs for the training', type=int, default=20)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'train_style_anomaly_detector','Training of the Adversarial Anomaly Detector model with a StyleGAN.')

    p.add_argument(     '--generatorDir',       help='Path of the generator h5 file', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the discriminator h5 file', default='')
    p.add_argument(     '--latentDim',          help='Latent space dimension of the GAN\'s generator', type=int, default=512)
    p.add_argument(     '--reconstructionError',help='Reconstruction error weight', type=float, default=0.90)
    p.add_argument(     '--dicriminatorError',  help='Discriminator error weight', type=float, default=0.10)
    p.add_argument(     '--epochs',             help='Number of epochs for the training', type=int, default=20)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'train_anomaly_grid_search','Training of the Adversarial Anomaly Detector classifier.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--normlaDataDir',      help='Path of the dataset with healthy samples', default='')
    p.add_argument(     '--anomalyDataDir',     help='Path of the dataset with anomal samples', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'train_anomaly_classifier','Training of the Adversarial Anomaly Detector classifier.')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=100)
    p.add_argument(     '--C',                  help='SVM C parameter', type=float, default=-1)
    p.add_argument(     '--gamma',              help='SVM gamma parameter', type=float, default=0.001)
    p.add_argument(     '--kernel',             help='SVM kernel parameter', default='rbf')
    p.add_argument(     '--degree',             help='SVM degree parameter', type=int, default=3)
    p.add_argument(     '--normlaDataDir',      help='Path of the dataset with healthy samples', default='')
    p.add_argument(     '--anomalyDataDir',     help='Path of the dataset with anomal samples', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline(sys.argv)

#----------------------------------------------------------------------------

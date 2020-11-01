""" Tool for use the model capabilities """

import sys
import argparse
import numpy as np

from models.AAD import AAD
from models.DCGAN import DCGAN
from models.StyleAAD2 import StyleAAD2
from models.stylegan import StyleGAN_G
from models.stylegan import StyleGAN_D
from matplotlib import pyplot
from tensorflow.keras.utils import Progbar
from utils.PreProcessing import load_test_data, load_labeled_data, postprocessing

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

    data = load_test_data(dataDir, 64, 'channels_first')

    anomaly_detector = StyleAAD2(style_gan_g, style_gan_d, resultsDir, latentDim)
    anomaly_detector.load(aadModelDir)
    anomaly_detector.plot_batch(data)

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

def evaluate_style_anomalies_in_dataset(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, dataDir, resultsDir):
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

    dataset = load_test_data(dataDir, 64, 'channels_first')

    normal, anomaly = anomaly_detector.evaluate_subset(dataset)
    anomaly_detector.t_sne_analysis(normal, anomaly, channel_format='channels_first')

def plot_style_mosaic(generatorDir, discriminatorDir, anomalyDetectorDir, SVCDir, ScalerDir, latentDim, dataDir, resultsDir):
    dimension = 64

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
    
    (dataset, rows, columns) = load_labeled_data(dataDir, dim=dimension, format='channels_first')

    output = np.zeros((int(rows * dimension), int(columns * dimension), 3))
    progress_bar = Progbar(target=len(dataset))

    for i, image_pack in enumerate(dataset):
        img = image_pack['img']
        row = image_pack['row']
        column = image_pack['column']

        # Perform prediction
        score, class_predicted, res = anomaly_detector.predict(np.asarray([img]))

        # data pos-processing
        img = (img*127.5)+127.5
        img = img.astype(np.uint8)
        img = img.transpose([1, 2, 0])

        reconstructed_img = res[0]
        reconstructed_img = reconstructed_img.transpose([1, 2, 0])

        # classify image
        if (class_predicted == 1):
            img = postprocessing(img, reconstructed_img)

        y = int(row * dimension)
        x = int(column * dimension)
        output[y: (y + dimension), x:(x + dimension)] = img

        progress_bar.update(i, values=[('score', score)])

    # save original image
    pyplot.figure(3, figsize=(3, 3))
    pyplot.axis('off')
    pyplot.imshow(output)
    pyplot.savefig(resultsDir + '/mosaic_image.pdf')

    pyplot.show()

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

    p = add_command(    'evaluate_style_anomalies_in_dataset', 'Evaluate the presence of anomalies in some dataset using the StyleGAN')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=512)
    p.add_argument(     '--dataDir',            help='Path of the dataset', default='')
    p.add_argument(     '--resultsDir',         help='Path where the results will be stored', default='')

    p = add_command(    'plot_style_mosaic', 'Plot mosaic image')

    p.add_argument(     '--generatorDir',       help='Path of the GAN\'s generator weights', default='')
    p.add_argument(     '--discriminatorDir',   help='Path of the GAN\'s discriminator weights', default='')
    p.add_argument(     '--anomalyDetectorDir', help='Path of the AnomalyDetector weights', default='')
    p.add_argument(     '--SVCDir',             help='Path of the Support Vector Machine weights', default='')
    p.add_argument(     '--ScalerDir',          help='Path of the Scaler weights', default='')
    p.add_argument(     '--latentDim',          help='Latent dimension of the GAN', type=int, default=512)
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

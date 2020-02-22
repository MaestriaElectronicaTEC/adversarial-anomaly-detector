""" Tool for creating the dataset """

import sys
import argparse

from dataset.imageSplit import process_image, process_images

#----------------------------------------------------------------------------

def cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating the dataset for the Adversarial Anomaly Detector.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'process_image',      'Process one image.',
                                              'process_image --imagePath path/to/image --distPath path/to/directory')
    p.add_argument(     '--imagePath',        help='Path of the image', default='')
    p.add_argument(     '--subsectionSize',   help='Size of each subsection H x W (Default (50, 50))', nargs='+', type=int)
    p.add_argument(     '--distPath',         help='Directory where the subsections will be stored', default='')
    p.add_argument(     '--overlap',          help='Activate the overlapping of the generated images', default=False, action='store_true')

    p = add_command(    'process_images',    'Process several images.',
                                             'process_image --imagePath path/to/images --distPath path/to/directory')
    p.add_argument(     '--imagesPath',       help='Path of the images directory', default='')
    p.add_argument(     '--subsectionSize',   help='Size of each subsection H x W (Default (50, 50))', nargs='+', type=int)
    p.add_argument(     '--distPath',         help='Directory where the subsections will be stored', default='')
    p.add_argument(     '--overlap',          help='Activate the overlapping of the generated images', default=False, action='store_true')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline(sys.argv)

#----------------------------------------------------------------------------
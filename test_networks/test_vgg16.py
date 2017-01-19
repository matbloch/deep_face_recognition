#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file",
        default="input_96.png",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "--output_file",
        default="network_output",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir, "models",
                "VGG_FACE16_deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir, "models",
                "VGG_FACE16.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_false',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_false',	# takes up to 10 times longe - network creates 10 different crops
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        # default=os.path.join(pycaffe_dir,
        #                      'caffe/imagenet/ilsvrc_2012_mean.npy'),
        default='',
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

	caffe.set_mode_cpu()
    # if args.gpu is True:
        # caffe.set_mode_gpu()
        # print("GPU mode")
    # else:
        # caffe.set_mode_cpu()
        # print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        print("Loading file: %s" % args.input_file)
        inputs = [caffe.io.load_image(args.input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    i_96 = [caffe.io.load_image("input_96.png")]
    i_128 = [caffe.io.load_image("input_128.png")]
    i_256 = [caffe.io.load_image("input_256.png")]

    start = time.time()
    predictions = classifier.predict(i_96, not args.center_only)
    print("--- 96x96px - predicted in %.2f s." % (time.time() - start))
    start = time.time()
    predictions = classifier.predict(i_128, not args.center_only)
    print("--- 128x128px - predicted in %.2f s." % (time.time() - start))
    start = time.time()
    predictions = classifier.predict(i_256, not args.center_only)
    print("--- 256x256px - predicted in %.2f s." % (time.time() - start))

    print "--- Feature vector dimension: {}".format(len(predictions[1]))
    # Save
    # print("Saving results into %s" % args.output_file)
    # np.save(args.output_file, predictions)


    # # Processing one image at a time, printint predictions and writing the vector to a file
    # with open(inputfile, 'r') as reader:
    #     with open(outputfile, 'w') as writer:
    #         writer.truncate()
    #         for image_path in reader:
    #             image_path = image_path.strip()
    #             input_image = caffe.io.load_image(image_path)
    #             prediction = net.predict([input_image], oversample=False)
    #             print os.path.basename(image_path), ' : ', labels[prediction[0].argmax()].strip(), ' (', prediction[0][
    #                 prediction[0].argmax()], ')'
    #             np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1, -1), fmt='%.8g')

if __name__ == '__main__':
    main(sys.argv)

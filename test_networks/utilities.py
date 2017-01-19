#!/usr/bin/env python
import numpy as np
import os
import sys
import glob
import time
import caffe
import pickle
from scipy import misc

current_dir = os.path.dirname(__file__)
VERBOSE = True


def load_pretrained_model(caffemodel, prototxt):
    model = os.path.join(current_dir, "models", caffemodel)
    model_definitions = os.path.join(current_dir, "models", prototxt)
    classifier = caffe.Classifier(model_definitions, model)  # rgb image assumed, convert to bgr
    return classifier


def load_images(img_folder):
    img_folder = os.path.join(current_dir, img_folder)
    if VERBOSE:
        print "--- Loading {} images from {}".format(len(os.listdir(img_folder)),img_folder)
    images = []
    for index, filename in enumerate(os.listdir(img_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images.append(misc.imread(img_folder + '/' + filename))
    return images


def calc_embeddings_from_imgs(clf, images):
    tot_files = len(images)
    timings = []
    embeddings = []
    for index, image in enumerate(images):
        print "--- Processing file {}/{}".format(index + 1, tot_files)
        start = time.time()
        caffe_out = clf.predict([image], True)
        timings.append(time.time() - start)
        embeddings.append(caffe_out[0])

    return embeddings, timings

def calc_embeddings(clf, img_folder):

    img_folder = os.path.join(current_dir, img_folder)
    tot_files = len(os.listdir(img_folder))
    tot_images = 0
    embeddings = []
    timings = []
    calc_start = time.time()

    print "--- starting to generate embeddings..."

    for index, filename in enumerate(os.listdir(img_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            tot_images += 1
            print "--- Processing file {}/{}".format(index+1, tot_files)

            # load image with scipy
            # image = misc.imread(path_in+file)

            # load image with caffe
            image = caffe.io.load_image(img_folder + '/' + filename)

            # calculate embedding with caffe
            start = time.time()
            caffe_out = clf.predict([image], True)
            timings.append(time.time()-start)
            embeddings.append(caffe_out[0])

            print(len(caffe_out[0]))
        else:
            continue

    print "--- embedding calculation took {} seconds | avg: {}".format(time.time()-calc_start, np.mean(timings))
    print "--- useable: {}/{} images".format(len(embeddings), tot_images)
    return embeddings, timings


def pkl_save(embeddings, filename):

    if not filename.endswith(".pkl"):
        filename += '.pkl'

    if VERBOSE:
        print("--- Saving database to '{}'".format(filename))

    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
        f.close()
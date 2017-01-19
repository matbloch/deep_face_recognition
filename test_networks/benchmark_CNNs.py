#!/usr/bin/env python
import numpy as np
import os
import sys
import argparse
import glob
import time
import caffe

import utilities as u


def main(argv):
    pass


def test_preprocessing():

    images = u.load_images("test_images")

    clf = u.load_pretrained_model(
          "disc_feature_approach_face_model.caffemodel",
          "disc_feature_approach_face_deploy.prototxt"
    )

    embeddings, timings = u.calc_embeddings_from_imgs(clf, images)

if __name__ == '__main__':
    # main(sys.argv)
    test_preprocessing()
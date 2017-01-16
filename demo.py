import time
import argparse
import numpy as np
import scipy.io as sio

import cPickle
import caffe
from scipy import misc

curr_file_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(fileDir, 'face_verification_experiment', 'model')
proto_dir = os.path.join(fileDir, 'face_verification_experiment', 'proto')


def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    f = open(filename, 'rb')
    header = f.read(4)
    f.close()
    if cmp(header, '\x50\x4b\x03\x04')==0:
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = cPickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
    fo.close()
    return dict

# ================================= #
#         Feature Extraction

def extract_features(img):
    network_proto_path = model_dir + ""
    network_model_path = proto_dir + "LightenedCNN_C_deploy.prototxt"

    print network_model_path

    # load network
    net = caffe.Classifier(network_proto_path, network_model_path)

    # parameters
    net.set_input_scale('data', 1)
    data_mean = None
    image_as_grey = False

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    # net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
    net.set_mean('data', data_mean)

    if not image_as_grey:
        net.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    # net.set_input_scale('data', 256)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_input_scale('data', 1)
    net.set_mode_cpu()



    # ----- test

    blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])

    layer_name = 'ip1'


    # blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
    shp = blobs[layer_name].shape
    print blobs['data'].shape

    batch_size = blobs['data'].shape[0]
    print blobs[layer_name].shape
    # print 'debug-------\nexit'

    # preprocessing
    if image_as_grey and img.shape[2] != 1:
        img = skimage.color.rgb2gray(img)
        img = img[:, :, np.newaxis]


    # prediction
    scores = net.predict([img], oversample=False)
    blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])

    return blobs

# ================================= #
#           Evaluation

def evaluation():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Image folder.", default="input")
    # parser.add_argument('--output', help="output filename", default="face_embeddings")

    # parse arguments
    args = parser.parse_args()

    # load image
    image = misc.imread("test.png")

    # extract feature vector
    feature_vector = extract_features(image)
    print len(feature_vector)

    pass

# ================================= #
#              Main

if __name__ == '__main__':
    evaluation()

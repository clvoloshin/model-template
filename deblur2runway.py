# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from PIL import Image
from options.test_options import TestOptions
from models.models import create_model
from data.data_loader import CreateDataLoader

from PIL import Image
import numpy as np
import torch

class DeblurHelper(object):

    def __init__(self, options):
        opt = TestOptions().parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        if options['use_single_gpu']:
            opt.gpu_ids = [0] # use first gpu
        else:
            opt.gpu_ids = [] # use cpu
        # opt.model = 'test'
        opt.dataset_mode = 'single'
        opt.learn_residual = True

        self.model = create_model(opt)
        pass

    # Deblur data image
    def run_on_input(self, data):
        '''
        data is an image
        '''
        data = np.array(data)
        data = np.rollaxis(data, 2) # put into [channels, height, width]
        # data = np.array(Image.fromarray(data).resize((256,256)))

        # convert to tensor and make data in [0,1]. Convert to double (expected)
        inp = torch.from_numpy(data[None,...]/255.).float()

        # get output
        out = self.model.netG.forward(inp)

        # convert to numpy
        output = out.data.numpy()
        output = np.rollaxis(output[0], 0, 3) # put back into [height, width, channels]
        output = np.array(output*255.).astype('uint8') # put back into uint8 format

        # make into image
        return Image.fromarray(output)

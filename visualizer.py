import visdom
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


class Visualizer(object):

    def __init__(self, env, **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.vis.close()

        self.iters = {}
        self.lines = {}

    def display_current_results(self, iters, x, name='train_loss'):
        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)

        self.vis.line(X=np.array(self.iters[name]),
                      Y=np.array(self.lines[name]),
                      win=name,
                      opts=dict(legend=[name], title=name))

    def curve(self, iters, x, name='train_loss'):
        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)

        self.vis.line(X=np.array(self.iters[name]),
                      Y=np.array(self.lines[name]),
                      win=name,
                      opts=dict(legend=[name], title=name))


    def img(self, name, img_, **kwargs):
        '''
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.detach().cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )
    

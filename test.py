"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from collections import Counter

import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def calculate_difference(visuals, original):
    fake_img = visuals['fake']
    simple_estimate = original['A'][0][0].reshape([1, 1, 256, 1024])
    real_img = original['B']
    fake = tensor2im(fake_img)
    real = tensor2im(real_img)
    est = tensor2im(simple_estimate)
    diff = (real%256-fake%256)
    diff2 = (real%256-est%256)
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    d_min = min(diff.flatten())
    d_max = max(diff.flatten())
    n_bins = d_max - d_min
    data = diff.flatten()
    data2 = diff2.flatten()
    mu, std = norm.fit(data)
    mu2, std2 = norm.fit(data2)
    n, bins, patches = plt.hist(data, bins=n_bins, density=True, alpha=0.6, color='b')
    plt.hist(data2,bins, density=True, alpha=0.6, color='r')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    p2 = norm.pdf(x, mu2, std2)
    plt.plot(x, p, 'k', linewidth=2)
    plt.plot(x, p2, '--', linewidth=2)
    title = f"Fit Values: {mu:.2f} +/- {std:.2f} vs {mu2:.2f} +/- {std2:.2f}"

    print(f'{std<std2:1d} pred: {std:.2f} estimate: {std2:.2f}')
    plt.title(title)

    plt.show()
    return std<std2
    # best fit of data
    (mu, sigma) = norm.fit(datos)

    # the histogram of the data
    n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    # plot
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.grid(True)

    plt.show()
    pass


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    retvals = Counter()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        retval = calculate_difference(visuals, data)
        retvals[retval] += 1
        save_images(webpage, visuals, img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize,
                    use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
    print(retvals)

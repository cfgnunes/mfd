import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def plot_image_list(image_list, title_list=None, ncols=2, scale_factor=2.0,
                    wspace=0.1, hspace=0.1, bgr_image=False):

    nimages = len(image_list)

    if nimages == 0:
        return

    if title_list and len(title_list) != len(image_list):
        raise Exception("'title_list' and 'image_list' must have same size.")

    ncols = int(np.min((ncols, nimages)))
    nrows = int(np.ceil(nimages / ncols))

    size_width = scale_factor * ncols
    size_height = scale_factor * nrows

    fig = plt.figure(figsize=(size_width, size_height))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,
                           wspace=wspace, hspace=hspace)

    for i, image in enumerate(image_list):
        ax = fig.add_subplot(gs[i])
        ax.axis('off')

        if bgr_image:
            image = __bgr2rgb(image)

        ax.imshow(image, cmap='gray')

        if title_list:
            ax.set_title(title_list[i])

    plt.show()


def plot_descriptor(descriptor, dict_names, figwidth=5):
    ncategories = len(dict_names)
    size = len(descriptor)
    indexes = range(0, size)

    fig, ax = plt.subplots()
    fig.set_figwidth(figwidth)

    bars = plt.bar(indexes, descriptor, width=1.0)

    for i, bar in enumerate(bars):
        orientation = i % ncategories

        color = dict_names[orientation][1]
        bar.set_facecolor(color)

        if i < ncategories:
            label = dict_names[orientation][0]
            bar.set_label(label)

    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
    ax.set_ylabel("Bin value")
    ax.set_title("Final descriptor (%i bins)" % size)
    plt.xticks(np.arange(0, size + 1, ncategories))

    plt.show()


def __bgr2rgb(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image[..., ::-1]
    return image

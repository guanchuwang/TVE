import torch
import matplotlib.pyplot as plt
import ipdb

def save_checkpoint(fname, **kwargs):

    checkpoint = {}

    for key, value in kwargs.items():
        checkpoint[key] = value
        # setattr(self, key, value)

    torch.save(checkpoint, fname)


def load_checkpoint(fname):

    return torch.load(fname)


def explanation_imshow(inputs, explanations, fnames=None, labels=None):

    batch_size = inputs.shape[0]

    for idx in range(batch_size):

        img = inputs[idx]
        input_showed = (img - img.min()) / (img.max() - img.min())
        input_showed = input_showed.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(input_showed)

        if explanations is not None:
            heatmap = explanations[idx]

            heatmap_large = torch.nn.functional.interpolate(heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1]),
                                                            size=(img.shape[1], img.shape[2])).view(img.shape[1], img.shape[2])

            plt.imshow(heatmap_large, alpha=0.4, cmap="jet")

            if labels is not None:
                label = labels[idx]
                plt.text(10, 20, label, size=35, color="green", weight="bold")

        fname = fnames[idx]
        # plt.axis('equal')
        plt.axis('off')
        # plt.subplots_adjust(left=0, bottom=0, top=1, right=1)
        plt.savefig(fname, bbox_inches='tight', pad_inches=-0.01)
        # plt.savefig(fname[:-4] + ".pdf", bbox_inches='tight', pad_inches=-0.01)
        plt.close()
        # print(fname)

    # ipdb.set_trace()

"""
A default plotting script.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    fig, axs = plt.subplots(
        1,
        1,
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        # hspace=0.,
        # wspace=0.,
    )
    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "test.pdf")

    plt.show()

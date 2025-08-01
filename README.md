# Installation

Clone this repository into a fresh Python 3.13 environment:

```
git clone https://github.com/PJ-Watson/niriss-tools.git
```

Install using:

```
python -m pip install ./niriss-tools
```

Navigate to line 177 of `site-packages/stsci/tools/stpyfits.py`, and change the `dtype` to `int`:

```
raw_data = np.zeros(shape=dims, dtype=int) + pixval
```

## FAQ

### Why can't I use older versions of Python?

You can, but I can't guarantee the package dependencies will resolve properly. If you manage to make it work, let me know and I'll update the installation instructions.

### Can I install this in an existing environment?

Probably. Although you may find that some existing packages have been reinstalled (almost certainly `grizli`, `drizzlepac`, and `photutils`).

### Why do I need to modify `stsci.tools`?

In Numpy 2.0, type promotion in this manner throws an exception. Until I (or someone else) checks that it isn't `grizli` or `drizzlepac` at fault here, this is the quickest solution.

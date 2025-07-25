"""
A default plotting script.
"""

import cdspyreadme
import plot_utils
from default_imports import *

if __name__ == "__main__":

    orig_path = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v4"
        / "catalogues"
        / "stage_7"
    )

    hlsp_name = "hlsp_glass-jwst_jwst_niriss_abell2744_v1_cat.fits"

    hlsp_tab = Table.read(orig_path / hlsp_name)
    # exit()

    tablemaker = cdspyreadme.CDSTablesMaker()
    table = tablemaker.addTable(hlsp_tab, name="catalogue.dat")

    table.get_column("RA").set_format("F8.6")
    table.get_column("DEC").set_format("F10.6")
    for c in hlsp_tab.colnames:
        if "flux" in c or "err" in c:
            table.get_column(c).set_format("E12.6")
            # hlsp_tab[c].format = ".5e"

    # tablemaker.toMRT()
    tablemaker.writeCDSTables()
    tablemaker.makeReadMe()

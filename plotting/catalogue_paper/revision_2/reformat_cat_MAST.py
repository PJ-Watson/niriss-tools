"""
A default plotting script.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    orig_path = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v4"
        / "catalogues"
        / "stage_7"
        / "a2744_spec_cat_niriss_20250508.fits"
    )

    hlsp_name = "hlsp_glass-jwst_jwst_niriss_abell2744_v1_cat.fits"

    with fits.open(orig_path) as orig_hdul:

        primary_header = orig_hdul[0].header

        primary_header["DATE-BEG"] = (
            "2022-06-28T22:04:38.674",
            "Date-time start of exposure",
        )
        primary_header["DATE-END"] = (
            "2023-07-07T14:29:36.453",
            "Date-time end of exposure",
        )
        primary_header["DOI"] = (
            "10.17909/kw3c-n857",
            "Digital Object Identifier for the HLSP data collection",
        )
        primary_header["HLSPID"] = (
            "GLASS-JWST",
            "The identifier (acronym) for this HLSP collection",
        )
        primary_header["HLSPLEAD"] = ("Tommaso Treu", "Full name of HLSP project lead")
        primary_header["HLSPNAME"] = (
            "The GLASS JWST Early Release Science Program",
            "Title for HLSP project, long form",
        )
        primary_header["HLSPTARG"] = (
            "Abell 2744",
            "Designation of the target(s) or field(s) for this HLSP",
        )
        primary_header["HLSPVER"] = ("1.0", "Version identifier for this HLSP product")
        primary_header["INSTRUME"] = ("NIRISS", "Instrument used to acquire the data")
        primary_header["LICENSE"] = ("CC BY 4.0", "License for use of these data")

        primary_header["LICENURL"] = (
            "https://creativecommons.org/licenses/by/4.0/",
            "Data license URL",
        )
        primary_header["MJD-BEG"] = (
            "59758.91989205544",
            "[d] exposure start time in MJD",
        )
        primary_header["MJD-END"] = (
            "60132.60389413194",
            "[d] exposure end time in MJD",
        )

        primary_header["OBSERVAT"] = (
            "JWST",
            "Observatory used to acquire the data",
        )
        primary_header["PROPOSID"] = (
            "1324",
            "JWST Proposal ID",
        )
        primary_header["TELESCOP"] = (
            "JWST",
            "Telescope used to acquire the data",
        )
        primary_header["TIMESYS"] = (
            "UTC",
            "Principal time system for time-related keywords",
        )
        primary_header["XPOSURE"] = (
            "65280",
            "Duration of exposure [s]",
        )
        primary_header["GRIZLIV"] = (
            "1.12.8",
            "Grizli version (Brammer+2019)",
        )
        primary_header["TF115WIM"] = (
            "5497.232",
            "Total image exposure time in F115W [s]",
        )
        primary_header["TF115WGR"] = (
            "20614.592",
            "Total grism exposure time in F115W [s]",
        )
        primary_header["TF150WIM"] = (
            "5497.232",
            "Total image exposure time in F150W [s]",
        )
        primary_header["TF150WGR"] = (
            "20614.592",
            "Total grism exposure time in F150W [s]",
        )
        primary_header["TF200WIM"] = (
            "2748.616",
            "Total image exposure time in F200W [s]",
        )
        primary_header["TF200WGR"] = (
            "10307.296",
            "Total grism exposure time in F200W [s]",
        )
        primary_header["CRDS_CTX"] = (
            "jwst_1173.pmap",
            "Operational context used",
        )
        primary_header["CONFFILE"] = (
            "*.221215.conf",
            "Grism trace configuration",
        )

        print(primary_header)

        full_data = Table(orig_hdul[1].data)

        full_data["RA"] *= u.deg
        full_data["DEC"] *= u.deg

        for c in full_data.colnames:
            if "flux" in c or "err" in c:
                full_data[c] *= u.erg / u.s / u.cm**2
                full_data[c].format = ".5e"

        orig_hdul[1] = fits.table_to_hdu(full_data)

        table_header = orig_hdul[1].header

        table_header.insert(8, ("EXTNAME", "NIRISS_Z_CAT"))
        table_header.insert(9, ("EXTVER", "GLASS-JWST ERS"))
        table_header.insert(
            10, ("RADESYS", "ICRS", "Equinox of the celestial coordinate system.")
        )

        orig_hdul.writeto(orig_path.parent / hlsp_name, overwrite=True)

    tab = Table.read(orig_path.parent / hlsp_name)
    tab.pprint()

    # fig, axs = plt.subplots(
    #     1,
    #     1,
    #     figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.8),
    #     constrained_layout=True,
    #     sharex=True,
    #     sharey=True,
    #     # hspace=0.,
    #     # wspace=0.,
    # )
    # # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    # fig.patch.set_alpha(0.0)

    # plt.savefig(save_dir / "test.pdf")

    # plt.show()

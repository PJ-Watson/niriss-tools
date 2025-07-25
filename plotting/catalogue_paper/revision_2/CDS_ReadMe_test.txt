              GLASS-JWST ERS: The NIRISS Spectroscopic Catalogue (Watson+, 2025)
================================================================================
The GLASS-JWST Early Release Science Programme: The NIRISS Spectroscopic
Catalogue
    Watson P. J., Vulcani B., Treu T., Roberts-Borsani G., Dalmasso N., He X.,
    Malkan M. A., Morishita T., Rojas Ruiz S., Zhang Y., Acharyya A.,
    Bergamini P., Bradač M., Fontana A., Grillo C., Jones T., Marchesini D.,
    Nanayakkara T., Pentericci L., Tubthong C., Wang X.
    < >
    =References ?
================================================================================
ADC_Keywords: Galaxy catalogs; Redshifts; Spectra, infrared; Surveys;
              Clusters, galaxy
Keywords: galaxies; James Webb; redshifts

Abstract:
    We release a spectroscopic redshift catalogue of sources in the Abell 2744
    cluster field, derived from JWST/NIRISS observations taken as part of the
    GLASS-JWST Early Release Science programme. We describe the data reduction,
    contamination modelling and source detection, as well as the data quality
    assessment, redshift determination and validation. The catalogue consists of
    354 secure and 134 tentative redshifts, of which 245 are new spectroscopic
    redshifts, spanning a range 0.1<=z<=8.2. These include 17 galaxies at the
    cluster redshift, one galaxy at z~8, and a triply-imaged galaxy at
    z=2.653{+/-}0.002. Comparing against galaxies with existing spectroscopic
    redshifts z_{spec}, we find a small offset of
    {Delta}z = z_{spec}-z_{niriss}/(1+z_{spec})=(1.3{+/-}1.6)x10^{-3}$. We also
    release a forced extraction tool ``pygrife'' and a visualisation tool
    ``pygcg'' to the community, to aid with the reduction and classification of
    grism data. This catalogue will enable future studies of the
    spatially-resolved properties of galaxies throughout cosmic noon, including
    dust attenuation and star formation. As the first exploitation of the
    catalogue, we discuss the spectroscopic confirmation of multiple image
    systems, and the identification of multiple overdensities at 1<z<2.7.

Description:
    As part of the GLASS-JWST ERS programme, we used JWST/NIRISS in the
    Wide-Field Slitless Spectroscopy (WFSS) mode, to conduct low-resolution
    spectroscopy in the Abell 2744 cluster field. The observing strategy called
    for approximately 18 hours of slitless spectroscopy and imaging, using the
    F115W, F150W, and F200W blocking filters to cover a wavelength range
    ~1.0--2.2 microns, at a spectral resolution R~150 (at 1.4 microns). The
    observations consisted of a single pointing, which covered an area of
    2.2'x2.2' centred on the core of the cluster.

    We reduced the data using version 1.12.8 of grizli (Brammer+2019), adopting
    the ``*221215.conf'' grism trace configuration files (Matharu+2022), and the
    STScI calibration pipeline, with CRDS context ``jwst_1173.pmap''. 3652
    sources were detected on the drizzled stacked direct images using SEP
    (Barbary+16). Through fitting the 2D spectra with grizli, and visually
    inspecting the results with multiple team members, we were able to determine
    the redshift for 488 galaxies in the field of view. Of these, 354 were
    flagged as ``secure'', and 134 as ``tentative''. We release here the
    catalogue containing these redshifts, and emission line fluxes measured from
    the 2D spectra, where available.

Objects:
    ----------------------------------------------------------
        RA   (ICRS)   DE        Designation(s)
    ----------------------------------------------------------
     00 14 20.02  -30 23 17.8   A2744 = ACO 2744
    ----------------------------------------------------------

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records  Explanations
--------------------------------------------------------------------------------
ReadMe          80        .  This file
catalog.dat    237     3652  The spectroscopic catalogue, containing the
                             locations of all NIRISS-detected sources, and the
                             best-fit redshifts (488) and emission line fluxes
                             where available.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: catalog.dat
--------------------------------------------------------------------------------
  Bytes   Format Units  Label               Explanations
--------------------------------------------------------------------------------
  1-  4   I4     ---    ID_NIRISS           [1/3895] Unique identifier for
                                            sources in this catalogue
  6- 13   F8.6   deg    RA                  [3.56/3.63] Right Ascension (J2000)
 15- 24   F10.6  deg    DEC                 [-30.42/-30.37] Declination (J2000)
 26- 33   F8.5   ---    Z_NIRISS            [0.11/8.2]? Best-fit grism redshift
 35- 35   I1     ---    Z_FLAG              Redshift quality flag (1)
 37- 39   F3.1   ---    F115W_072.0_QUALITY  Quality flag for the dispersed 2D
                                             spectra using the F115W filter,
                                             and a position angle of 72.0
                                             degrees (2)
 41- 43   F3.1   ---    F115W_341.0_QUALITY  Quality flag for the dispersed 2D
                                             spectra using the F115W filter,
                                             and a position angle of 341.0
                                             degrees (2)
 45- 47   F3.1   ---    F150W_072.0_QUALITY  Quality flag for the dispersed 2D
                                             spectra using the F150W filter,
                                             and a position angle of 72.0
                                             degrees (2)
 49- 51   F3.1   ---    F150W_341.0_QUALITY  Quality flag for the dispersed 2D
                                             spectra using the F150W filter,
                                             and a position angle of 341.0
                                             degrees (2)
 53- 55   F3.1   ---    F200W_072.0_QUALITY  Quality flag for the dispersed 2D
                                             spectra using the F200W filter,
                                             and a position angle of 72.0
                                             degrees (2)
 57- 59   F3.1   ---    F200W_341.0_QUALITY  Quality flag for the dispersed 2D
                                             spectra using the F200W filter,
                                             and a position angle of 341.0
                                             degrees (2)
 61- 61   I1     ---    N_LINES              The number of emission lines for
                                             which it was possible to measure
                                             a flux, from the set {
                                             [OII]-3727,3729; H{beta};
                                             [OIII]-4959,5007; H{alpha};
                                             [SII]-6716,6731; [SIII]-9068,9531}
 63- 81   A19    ---    NAME_LINES          ? The semicolon separated names of
                                            the measured emission lines
 83- 94   E12.6  mW/m2  flux_OII            The [OII]-3727,3729 emission line
                                            flux
 96-107   E12.6  mW/m2  err_OII             The [OII]-3727,3729 1{sigma} flux
                                            uncertainties
109-120   E12.6  mW/m2  flux_Hb             The H{beta} emission line flux
122-133   E12.6  mW/m2  err_Hb              The H{beta} 1{sigma} flux
                                            uncertainties
135-146   E12.6  mW/m2  flux_OIII           The [OIII]-4959,5007 emission line
                                            flux
148-159   E12.6  mW/m2  err_OIII            The [OIII]-4959,5007 1{sigma} flux
                                            uncertainties
161-172   E12.6  mW/m2  flux_Ha             The H{alpha} emission line flux
174-185   E12.6  mW/m2  err_Ha              The H{alpha} 1{sigma} flux
                                            uncertainties
187-198   E12.6  mW/m2  flux_SII            The [SII]-6716,6731 emission line
                                            flux
200-211   E12.6  mW/m2  err_SII             The [SII]-6716,6731 1{sigma} flux
                                            uncertainties
213-224   E12.6  mW/m2  flux_SIII           The [SIII]-9068,9531 emission line
                                            flux
226-237   E12.6  mW/m2  err_SIII            The [SIII]-9068,9531 1{sigma} flux
                                            uncertainties
--------------------------------------------------------------------------------
Note (1):
    0 = source not extracted
    1 = source rejected on initial inspection
    2 = undetermined redshift
    3 = tentative redshift
    4 = secure redshift
Note (2):
    0 = unusable data
    1 = poor quality
    2 = good quality

See also:
  J/ApJ/812/114 : Grism Lens-Amplified Survey from Space (GLASS) (Treu+, 2015)
  J/ApJ/940/L52 : Early results from GLASS-JWST. VI. NIRISS WFFS (Boyett+, 2022)
  J/ApJ/952/20  : GLASS-JWST ERS Prog. II. ACO 2744 (Paris+, 2023)
  J/ApJS/270/12 : UNCOVER SPS cat. for robust sources up to z∼15 (Wang+, 2024)
  J/ApJS/270/7  : UNCOVER phot. catalog of A2744 with HST+JWST (Weaver+, 2024)
  J/A+A/691/A240: ASTRODEEP-JWST photometry and redshifts (Merlin+, 2024)

References:
    Brammer G.               Grizli         2019ascl.soft05001B
    Matharu et al.           Grism Config.  DOI: 10.5281/zenodo.7628094
    Barbary et al.           SEP            2016zndo....159035B
    Roberts-Borsani et al.   Paper I.       2022ApJ...938L..13R
    Merlin et al.            Paper II.      2022ApJ...938L..14M
    Castellano et al.        Paper III.     2022ApJ...938L..15C
    Wang et al.              Paper IV.      2022ApJ...938L..16W
    Yang et al.              Paper V.       2022ApJ...938L..17Y
    Vanzella et al.          Paper VII.     2022ApJ...940L..53V
    Chen et al.              Paper VIII.    2022ApJ...940L..53V
    Marchesini et al.        Paper IX.      2023ApJ...942L..25M
    Leethochawalit et al.    Paper X.       2023ApJ...942L..26L
    Santini et al.           Paper XI.      2023ApJ...942L..27S
    Treu et al.              Paper XII.     2023ApJ...942L..28T
    Nonino et al.            Paper XIII.    2023ApJ...942L..29N
    Morishita et al.         Paper XIV.     2023ApJ...947L..24M
    Glazebrook et al.        Paper XV.      2023ApJ...947L..25G
    Nanayakkara et al.       Paper XVI.     2023ApJ...947L..26N
    Dressler et al.          Paper XVII.    2023ApJ...947L..27D
    Jacobs et al.            Paper XVIII.   2023ApJ...948L..13J
    Castellano et al.        Paper XIX.     2023ApJ...948L..14C
    Vulcani et al.           Paper XX.      2023ApJ...948L..15V
    Jones et al.             Paper XXI.     2023ApJ...951L..17J
    Roy et al.               Paper XXII.    2023ApJ...952L..14R
    Prieto-Lyon et al.       Paper XXIII.   2023ApJ...956..136P
    He et al.                Paper XXIV.    2024ApJ...960L..13H
    Sijia et al.             Paper XXV.     2025ApJ...979L..13L
    Treu et al.              Release I      2022ApJ...935..110T
    Paris et al.             Release II     2023ApJ...952...20P
    Bergamini et al.         Release III    2023ApJ...952...84B
    Mascia et al.            Release IV     2024A&A...690A...2M

================================================================================
(End)                                 (prepared by Peter J. Watson  / pyreadme )

"""Template script for generating a BrainGlobe atlas.

Use this script as a starting point to package a new BrainGlobe atlas by
filling in the required functions and metadata.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import pooch

from brainglobe_utils.IO.image import load_any
from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

ATLAS_NAME = "duke_dev_rat"
CITATION = "https://doi.org/10.1016/j.neuroimage.2013.01.017"
SPECIES = "Rattus norvegicus"
ATLAS_LINK = "https://civmvoxport.vm.duke.edu/voxbase/studyhome.php?studyid=208"
ORIENTATION = "asr"

ROOT_ID = 999
RESOLUTION = 25
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22891"
ANNOTATION_URL = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22940"
LABELS_URL = "https://civmvoxport.vm.duke.edu/voxbase/get_attachment.php?attachmentID=402"

ref00 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22945"
ref02 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22912"
ref04 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22909"
ref08 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22906"
ref12 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22903"
ref18 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22900"
ref24 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22897"
ref40 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22893"
ref80 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22891"

ann00 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22916"
ann02 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22919"
ann04 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22920"
ann08 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22925"
ann12 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=23604"
ann18 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22931"
ann24 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22934"
ann40 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22937"
ann80 = "https://civmvoxport.vm.duke.edu/voxbase/downloaddataset.php?stackID=22940"


BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

TIMEPOINTS = ["00", "02", "04", "08", "12", "18", "24", "40", "80"]

REFERENCE_FNAMES = {age: "p" + age + "_average_gre.nii" for age in TIMEPOINTS}
ANNOTATION_FNAMES = {age: "pnd" + age + "_average_labels.nii" for age in TIMEPOINTS}
LABELS_FNAME = "Developmental_labels_lookup.txt"

def pooch_init(download_dir_path: Path, timepoints: list[str]) -> pooch.Pooch:
    """Initialize Pooch for downloading atlas data.

    Parameters
    ----------
    download_dir_path : Path
        Path to the directory where data will be downloaded.
    timepoints : list[str]
        List of timepoints for which data archives are expected.

    Returns
    -------
    pooch.Pooch
        Initialized Pooch instance.
    """
    utils.check_internet_connection()
    
    keys = list(REFERENCE_FNAMES.values()) + list(ANNOTATION_FNAMES.values()) + [LABELS_FNAME]
    empty_registry = {key: None for key in keys}
    
    p = pooch.create(
        path=download_dir_path,
        base_url="",
        registry=empty_registry,
    )
    p.load_registry(Path(__file__).parent / "hashes" / (ATLAS_NAME + ".txt"))
    return p

def fetch_animal(pooch_: pooch.Pooch, age: str):
    """Fetch annotation and reference volumes for a specific age.

    Parameters
    ----------
    pooch_ : pooch.Pooch
        The initialized Pooch instance.
    age : str
        The age timepoint (e.g., "00", "24").

    Returns
    -------
    tuple
        A tuple containing:
        - annotations (np.ndarray): The annotation volume.
        - reference (np.ndarray): The reference volume (scaled to uint16).

    Raises
    ------
    AssertionError
        If an unknown age timepoint is provided.
    """
    assert age in TIMEPOINTS, f"Unknown age timepoint: '{age}'"
    
    fetched_reference = pooch_.fetch(
        REFERENCE_FNAMES[age],
        progressbar=True,
    )
    
    fetched_annotation = pooch_.fetch(
        ANNOTATION_FNAMES[age], 
        progressbar=True,
    )
    
    reference = load_any(fetched_reference, as_numpy = True)
    annotations = load_any(fetched_annotation, as_numpy = True)
    '''dmin = np.min(reference)
    dmax = np.max(reference)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference = (reference - dmin) * dscale
    reference = reference.astype(np.uint16)'''
    return annotations, reference

def fetch_ontology(pooch_: pooch.Pooch):
    """Fetch and parse the ontology (structure tree) from an Excel file.

    Parameters
    ----------
    pooch_ : pooch.Pooch
        The initialized Pooch instance.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a brain
        structure with its properties (id, acronym, name, path, RGB color).
    """
    devccfv1_path = pooch_.fetch(
        "DevCCFv1_OntologyStructure.xlsx", progressbar=True
    )
    xl = pd.ExcelFile(devccfv1_path)
    # xl.sheet_names # it has two excel sheets
    # 'DevCCFv1_Ontology', 'README'
    df = xl.parse("DevCCFv1_Ontology", header=1)
    df = df[["Acronym", "ID16", "Name", "Structure ID Path16", "R", "G", "B"]]
    df.rename(
        columns={
            "Acronym": "acronym",
            "ID16": "id",
            "Name": "name",
            "Structure ID Path16": "structure_id_path",
            "R": "r",
            "G": "g",
            "B": "b",
        },
        inplace=True,
    )
    structures = list(df.to_dict(orient="index").values())
    for structure in structures:
        if structure["acronym"] == "mouse":
            structure["acronym"] = "root"
        structure_path = structure["structure_id_path"]
        structure["structure_id_path"] = [
            int(id) for id in structure_path.strip("/").split("/")
        ]
        structure["rgb_triplet"] = [
            structure["r"],
            structure["g"],
            structure["b"],
        ]
        del structure["r"]
        del structure["g"]
        del structure["b"]
    return structures

def download_resources():
    
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    needs_download = (
        (not reference_path.exists())
        or (not annotation_path.exists())
        or (not labels_path.exists())
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(reference_path):
        pooch.retrieve(
            url=REFERENCE_URL,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
        )

    if should_fetch(labels_path):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=LABELS_FNAME,
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference = None
    annotation = None
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    Return a list of dictionaries with information about the atlas.

    Returns a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

    The expected format for each dictionary is:

    .. code-block:: python

        {
            "id": int,
            "name": str,
            "acronym": str,
            "structure_id_path": list[int],
            "rgb_triplet": list[int, int, int],
        }

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
    """
    return None


def retrieve_or_construct_meshes():
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    If the atlas is packaged with mesh files, download and use them. Otherwise,
    construct the meshes using available helper functions.

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to the
        corresponding mesh files.
    """
    meshes_dict = {}
    return meshes_dict


def retrieve_additional_references():
    """
    Return a dictionary of additional reference images.

    This function should be edited only if the atlas includes additional
    reference images. The dictionary should map the name of each additional
    reference image to its corresponding image stack data.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    additional_references = {}
    return additional_references


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    if RESOLUTION is None:
        raise ValueError("RESOLUTION must be set before running this script.")

    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(bg_root_dir.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {bg_root_dir}. "
        )
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes()

    '''output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RESOLUTION,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )'''

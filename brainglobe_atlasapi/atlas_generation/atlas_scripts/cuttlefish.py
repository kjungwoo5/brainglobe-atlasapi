__version__ = "0"

import csv
import glob as glob
from pathlib import Path

import pooch
from brainglobe_utils.IO.image import load

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data


def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i : i + 2], 16)
        rgb.append(decimal)

    return rgb


def create_atlas(working_dir, resolution):
    ATLAS_NAME = "columbia_cuttlefish"
    SPECIES = "Sepia bandensis"
    ATLAS_LINK = "https://www.cuttlebase.org/"
    CITATION = (
        "Montague et al, 2023, https://doi.org/10.1016/j.cub.2023.06.007"
    )
    ORIENTATION = "srp"
    ATLAS_PACKAGER = "Jung Woo Kim"
    ADDITIONAL_METADATA = {}

    HIERARCHY_FILE_URL = "https://raw.githubusercontent.com/noisyneuron/cuttlebase-util/main/data/brain-hierarchy.csv"  # noqa E501
    TEMPLATE_URL = r"https://www.dropbox.com/scl/fo/fz8gnpt4xqduf0dnmgrat/ABflM0-v-b4_2WthGaeYM4s/Averaged%2C%20template%20brain/2023_FINAL-Cuttlebase_warped_template.nii.gz?rlkey=eklemeh57slu7v6j1gphqup4z&dl=1"  # noqa E501
    ANNOTATION_URL = r"https://www.dropbox.com/scl/fo/fz8gnpt4xqduf0dnmgrat/ALfSeAj81IM0v56bEeoTfUQ/Averaged%2C%20template%20brain/2023_FINAL-Cuttlebase_warped_template_lobe-labels.nii.seg.nrrd?rlkey=eklemeh57slu7v6j1gphqup4z&dl=1"  # noqa E501

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    # download hierarchy files
    utils.check_internet_connection()
    hierarchy_path = pooch.retrieve(
        HIERARCHY_FILE_URL,
        known_hash="023418e626bdefbd177d4bb8c08661bd63a95ccff47720e64bb7a71546935b77",
        progressbar=True,
    )

    # import cuttlefish .nrrd file
    annotation_path = pooch.retrieve(
        ANNOTATION_URL,
        known_hash="768973251b179902ab48499093a4cc870cb6507c09ce46ff76b8203daf243f82",
        progressbar=True,
    )

    import nrrd

    # process brain annotation file. There are a total of 70 segments.
    print("Processing brain annotations:")
    readdata, header = nrrd.read(annotation_path)

    # Extract annotation mapping information from nrrd headers,
    # to be applied to hierarchy file later.
    mapping = []
    for n in range(0, 70):
        mapping.append(
            {
                "color": header[f"Segment{n}_Color"],
                "id": header[f"Segment{n}_LabelValue"],
                "acronym": header[f"Segment{n}_Name"],
            }
        )

    # convert the color information stored as a string of 3 RGB floats
    # into a list of 3 RGB integers from 0 to 255.
    for index, Map in enumerate(mapping):
        mapping[index]["color"] = Map["color"].split(" ")
        mapping[index]["color"] = list(map(float, mapping[index]["color"]))
        mapping[index]["color"] = [
            int(255 * x) for x in mapping[index]["color"]
        ]

    # print(mapping)
    # df = pd.DataFrame(mapping)
    # df.to_csv('mappingtest.csv')

    # create dictionaries
    print("Creating structure tree")
    with open(
        hierarchy_path, mode="r", encoding="utf-8-sig"
    ) as cuttlefish_file:
        cuttlefish_dict_reader = csv.DictReader(cuttlefish_file)

        # empty list to populate with dictionaries
        hierarchy = []

        # parse through csv file and populate hierarchy list
        for row in cuttlefish_dict_reader:
            if row["hasSides"] == "Y":
                leftSide = dict(row)
                leftSide["abbreviation"] = leftSide["abbreviation"] + "l"
                leftSide["name"] = leftSide["name"] + " (left)"

                rightSide = dict(row)
                rightSide["abbreviation"] = rightSide["abbreviation"] + "r"
                rightSide["name"] = rightSide["name"] + " (right)"

                hierarchy.append(leftSide)
                hierarchy.append(rightSide)
            else:
                hierarchy.append(row)

    # use layers to give IDs to regions which do not have existing IDs.
    layer1 = 100
    layer2 = 200
    # remove 'hasSides' and 'function' keys,
    # reorder and rename the remaining keys
    for i in range(0, len(hierarchy)):
        hierarchy[i]["acronym"] = hierarchy[i].pop("abbreviation")
        hierarchy[i].pop("hasSides")
        hierarchy[i].pop("function")
        hierarchy[i]["structure_id_path"] = list(
            (map(int, (hierarchy[i]["index"].split("-"))))
        )
        hierarchy[i]["structure_id_path"].insert(0, 999)
        hierarchy[i].pop("index")
        if (
            len(hierarchy[i]["structure_id_path"]) < 4
            and hierarchy[i]["structure_id_path"][-2] != 3
        ):
            if len(hierarchy[i]["structure_id_path"]) == 3:
                hierarchy[i]["id"] = layer2
                layer2 += 1
            elif len(hierarchy[i]["structure_id_path"]) == 2:
                hierarchy[i]["id"] = layer1
                layer1 += 1
        if hierarchy[i]["acronym"] == "SB":
            hierarchy[i]["id"] = 71
        elif hierarchy[i]["acronym"] == "IB":
            hierarchy[i]["id"] = 72

    # remove erroneous key for the VS region
    # (error due to commas being included in the 'function' column)
    hierarchy[-3].pop(None)
    hierarchy[-4].pop(None)

    # add the 'root' structure
    hierarchy.append(
        {
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "id": 999,
        }
    )

    # apply colour and id map to each region
    for index, region in enumerate(hierarchy):
        for Map in mapping:
            if region["acronym"] == Map["acronym"]:
                hierarchy[index]["rgb_triplet"] = Map["color"]
                hierarchy[index]["id"] = int(Map["id"])

    # amend each region's structure_id_path by iterating through entire list,
    # and replacing dummy values with actual ID values.
    for i in range(0, len(hierarchy)):
        if len(hierarchy[i]["structure_id_path"]) == 2:
            hierarchy[i]["structure_id_path"][1] = hierarchy[i]["id"]
            len2_shortest_index = i

        elif len(hierarchy[i]["structure_id_path"]) == 3:
            hierarchy[i]["structure_id_path"][1] = hierarchy[
                len2_shortest_index
            ]["id"]
            hierarchy[i]["structure_id_path"][2] = hierarchy[i]["id"]
            len3_shortest_index = i

        elif len(hierarchy[i]["structure_id_path"]) == 4:
            hierarchy[i]["structure_id_path"][1] = hierarchy[
                len2_shortest_index
            ]["id"]
            hierarchy[i]["structure_id_path"][2] = hierarchy[
                len3_shortest_index
            ]["id"]
            hierarchy[i]["structure_id_path"][3] = hierarchy[i]["id"]

    # original atlas does not give colours to some regions, so we give
    # random RGB triplets to regions without specified RGB triplet values
    random_rgb_triplets = [
        [156, 23, 189],
        [45, 178, 75],
        [231, 98, 50],
        [12, 200, 155],
        [87, 34, 255],
        [190, 145, 66],
        [64, 199, 225],
        [255, 120, 5],
        [10, 45, 90],
        [145, 222, 33],
        [35, 167, 204],
        [76, 0, 89],
        [27, 237, 236],
        [255, 255, 255],
    ]

    n = 0
    for index, region in enumerate(hierarchy):
        if "rgb_triplet" not in region:
            hierarchy[index]["rgb_triplet"] = random_rgb_triplets[n]
            n = n + 1

    # give filler acronyms for regions without specified acronyms
    missing_acronyms = [
        "SpEM",
        "VLC",
        "BsLC",
        "SbEM",
        "PLC",
        "McLC",
        "PvLC",
        "BLC",
        "PeM",
        "OTC",
        "NF",
    ]
    n = 0
    for index, region in enumerate(hierarchy):
        if hierarchy[index]["acronym"] == "":
            hierarchy[index]["acronym"] = missing_acronyms[n]
            n = n + 1

    # import cuttlefish .nii file
    template_path = pooch.retrieve(
        TEMPLATE_URL,
        known_hash="195125305a11abe6786be1b32830a8aed1bc8f68948ad53fa84bf74efe7cbe9c",  # noqa E501
        progressbar=True,
    )

    # process brain template MRI file
    print("Processing brain template:")
    brain_template = load.load_nii(template_path, as_array=True)

    # check the transformed version of the hierarchy.csv file
    print(hierarchy)
    # df = pd.DataFrame(hierarchy)
    # df.to_csv('hierarchy_test.csv')

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=resolution,
        orientation=ORIENTATION,
        root_id=999,
        reference_stack=brain_template,
        annotation_stack=readdata,
        structures_list=hierarchy,
        meshes_dict={},
        scale_meshes=True,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,
        additional_references={},
    )

    return output_filename


if __name__ == "__main__":
    res = 2, 2, 2
    home = str(Path.home())
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, res)

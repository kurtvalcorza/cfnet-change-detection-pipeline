"""Labelled-pair dataset contract for adapting the change detector: the pinned LEVIR-CD sample, roles from the
dataset's own splits, BYOD loaders and sample export.

The default dataset is **real**: 64 labelled 256 × 256 crops of LEVIR-CD (Chen and Shi, 2020) — pairs of 0.5 m
Google Earth patches of Texas cities taken 5–14 years apart with building-change labels — as the CFNet authors
processed and mirrored them on the Hugging Face Hub (`wifibk/CFNet_Datasets`, `LEVIR-CD-processed.tar.gz`: 25
overlapping 256 × 256 crops of every 1024 × 1024 pair, 11,125 / 1,600 / 3,200 crops in the train / val / test
folders). The 64 crops were drawn on 2026-09-20 with a fixed seed from the crops whose label is a clean 0 / 255
mask with at least 3 % change — one crop per source pair, from 32 training pairs, 8 validation pairs and 24 test
pairs — so the roles are the dataset's published splits (the checkpoint was trained on the training split,
selected on the validation split and reported on the test split). The tarball is pinned by byte size and SHA-256,
each pinned member is pinned again by size and SHA-256 and extracted **without** `extractall` into the cache, and
everything else in the archive is left alone. The repository redistributes none of the images.

**LEVIR-CD's terms:** "All images and annotations in LEVIR-CD can only be used for academic purposes, but are
prohibited for any commercial use", and the imagery is subject to Google Earth's terms of use. The tutorial fetches
the authors' mirror at run time for that academic purpose and nothing else; a commercial deployment must bring its
own labelled pairs.

A record is ``{id, before, after, label}``: two (H, W, 3) uint8 RGB images (or PNG / JPEG paths) and an (H, W) mask
with 0 = unchanged, 1 = changed, -1 = no data (or a PNG path with 0 / 255).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import (
    MIN_RECORDS,
    MODEL_ID,
    SAMPLE_SIZE,
    check_record,
    pair_digest,
    read_image,
    read_mask,
    validate_dataset,
)

CORPUS_NAME = "LEVIR-CD building-change crops (Chen and Shi, 2020), the CFNet authors' processed mirror"
CORPUS_RELEASE = (
    "Hugging Face dataset wifibk/CFNet_Datasets, LEVIR-CD-processed.tar.gz, 64 crops selected 2026-09-20 from the "
    "dataset's own splits"
)
CORPUS_LICENSE = "LEVIR-CD: academic purposes only, commercial use prohibited; imagery subject to Google Earth's terms of use"
DATASET_ID = "wifibk/CFNet_Datasets"
DATASET_REVISION = "ba68aa9a54ae15fe32ce9b02c380eb384fae528e"
CORPUS_BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
TAR_NAME = "LEVIR-CD-processed.tar.gz"
TAR_BYTES = 3_831_872_824
TAR_SHA256 = "6515dd451c159b9ed5bd53b3fb6e15188dd114b296ff9169ca21fbcabcd1d109"
CORPUS_BYTES = 16_921_921  # the 192 pinned members, uncompressed
DEFAULT_CACHE_DIR = Path("weights") / "levir-cd"
ROLES = ("train", "validation", "test")
# (crop key, role = the dataset split, source pair id, before member, bytes, sha256, after member, bytes, sha256,
#  label member, bytes, sha256) — member paths are relative to the tarball root (the archive prefixes them with `./`)
SAMPLE_RECORDS: tuple[tuple[str, str, int, str, int, str, str, int, str, str, int, str], ...] = (
    (
        "train_2_13",
        "train",
        2,
        "LEVIR-CD-processed/train/A/train_2_13.png",
        131373,
        "9101cf4d881d921e4fbb97f7ab17fc684fe8f5dbd15f7fc9448d973e405b1dc8",
        "LEVIR-CD-processed/train/B/train_2_13.png",
        139058,
        "de76943a524787fbf19264c3822e7c393f605d138af8c005bbcaea72a0d80b95",
        "LEVIR-CD-processed/train/label/train_2_13.png",
        2017,
        "a6f218f5b98517e70d07bb7d8a42776fd446c634234b02485e6444ec4de9fa5e",
    ),
    (
        "train_3_8",
        "train",
        3,
        "LEVIR-CD-processed/train/A/train_3_8.png",
        146642,
        "56fea1f62235b8a03594a09406658c5382a7b451bd396d51dd5bacceda3f3332",
        "LEVIR-CD-processed/train/B/train_3_8.png",
        135129,
        "4866e3f40162bcd083b85825881a7ad3dbb6cba99b0e7c4b47779f052a94443d",
        "LEVIR-CD-processed/train/label/train_3_8.png",
        1122,
        "7b4a6645859f2944003d321079023f1b212b50ca16648e2edb093db3e8723400",
    ),
    (
        "train_7_18",
        "train",
        7,
        "LEVIR-CD-processed/train/A/train_7_18.png",
        148943,
        "122222370c3e0d64245db936f7ef9c49e908672ef1e47db472800d946d3fda4d",
        "LEVIR-CD-processed/train/B/train_7_18.png",
        127318,
        "8b716e17736e18412e4666bea6a1313f3759c87014e7a027d2fcb0287f236f58",
        "LEVIR-CD-processed/train/label/train_7_18.png",
        490,
        "0ab9e39abdd27c0b1ebfd982de071901f6ba91569e64a583fbe97b9c634f0b0f",
    ),
    (
        "train_26_9",
        "train",
        26,
        "LEVIR-CD-processed/train/A/train_26_9.png",
        162404,
        "53cb963c8c8d62307394847733c3ecf93f6811dc780547cb6ea269f81ac45a45",
        "LEVIR-CD-processed/train/B/train_26_9.png",
        145061,
        "0eb5b8b6a1dc34f66f81bb3110cae07779d5defc3bdae6240b836f16a81cd76a",
        "LEVIR-CD-processed/train/label/train_26_9.png",
        1189,
        "c6f894e6fd7ea55ccd25cfdfc8175ef36ce12faec43e657d2848d84d5deefaa6",
    ),
    (
        "train_30_22",
        "train",
        30,
        "LEVIR-CD-processed/train/A/train_30_22.png",
        149046,
        "e3a0de96355906af020a8f9aca527f2fbb37080088f18f1074456ae489351425",
        "LEVIR-CD-processed/train/B/train_30_22.png",
        131261,
        "24c021cc1efd9555587a1ee95ff2a570c56c78c2e530f480ec13d060165253af",
        "LEVIR-CD-processed/train/label/train_30_22.png",
        560,
        "2f1a711ad3ad1d0affc18f2e49ca87c793889fcc8dfe721909aadfc15cd1d45b",
    ),
    (
        "train_37_23",
        "train",
        37,
        "LEVIR-CD-processed/train/A/train_37_23.png",
        101625,
        "7da15823b2769bd9b4fdcb33024649b6c0e0b41b6d39e7ff9e6b3f44fa4b4514",
        "LEVIR-CD-processed/train/B/train_37_23.png",
        134786,
        "26249b35b50114abfb96342b49118195cb2bcc6b7958f6203a65f15744a13cde",
        "LEVIR-CD-processed/train/label/train_37_23.png",
        583,
        "5c952c2a100568538d702f1b5d581d7493ed26815bb623a7ee7639c92bad4567",
    ),
    (
        "train_42_0",
        "train",
        42,
        "LEVIR-CD-processed/train/A/train_42_0.png",
        118551,
        "746181a35206c07495c54349209dba9912da84ea8c200de98f417a20c0e45d62",
        "LEVIR-CD-processed/train/B/train_42_0.png",
        143430,
        "6afc6b6f8c74858608368d7506301f37ef4be4e4840f2a18247eabc2567fa5cd",
        "LEVIR-CD-processed/train/label/train_42_0.png",
        513,
        "c2c52111e9d56b798d8d0086baf1ecc2c3e172252e9ee08a4abf95d3e3aa4973",
    ),
    (
        "train_51_15",
        "train",
        51,
        "LEVIR-CD-processed/train/A/train_51_15.png",
        162074,
        "f91d7a962b04cf6739d31d8dbb6d5d9d7ae00bb7c0bb519f8165bfae8cbcd095",
        "LEVIR-CD-processed/train/B/train_51_15.png",
        130651,
        "995c408511588fa2db0cf8d97eec714dbf8fd0ea6e315412fc47b73eef35a601",
        "LEVIR-CD-processed/train/label/train_51_15.png",
        895,
        "9d1a01fc3e786e59d832d6e8250ef9dacffeb46cf1ba25bfb246a9d55a50a92d",
    ),
    (
        "train_84_11",
        "train",
        84,
        "LEVIR-CD-processed/train/A/train_84_11.png",
        170532,
        "daa84ea86ac5f2654f3954852822475bc57030b2501004b68a8880b832c19009",
        "LEVIR-CD-processed/train/B/train_84_11.png",
        136411,
        "bfd1adca5316e48bd315b634c1875daea47d272c4c740a87446e8d5a6460e71d",
        "LEVIR-CD-processed/train/label/train_84_11.png",
        553,
        "49fe1cb17c960ae494c5f25f9bc779aa37cd60bde1681a3c01050edb7ca5f10e",
    ),
    (
        "train_95_24",
        "train",
        95,
        "LEVIR-CD-processed/train/A/train_95_24.png",
        165684,
        "8a48cca61035b190374fbd2b1afb53229bad49686195be646707039d0c924871",
        "LEVIR-CD-processed/train/B/train_95_24.png",
        135304,
        "fc0f940d25ead08e4b3659616b002caee07e42261b1fba1616d3077e8fa4b319",
        "LEVIR-CD-processed/train/label/train_95_24.png",
        1178,
        "edebce2ef7eb9c4d394dd35e8e440e57fce547aeb7a8e3f389df522c01818acf",
    ),
    (
        "train_111_0",
        "train",
        111,
        "LEVIR-CD-processed/train/A/train_111_0.png",
        177762,
        "a7af358d0b07bf359836472a62aee644ff915cbd402a2c0a6fda6904f10a2176",
        "LEVIR-CD-processed/train/B/train_111_0.png",
        142613,
        "0f8742754a87dfba3f50f77effe95ad9b32524550f5784ad20ea26084814a2ff",
        "LEVIR-CD-processed/train/label/train_111_0.png",
        2800,
        "3c26e193ce0a1efbd6b1ea2b033e8a3d31f170b1f787698254065b9eb9df86e6",
    ),
    (
        "train_118_6",
        "train",
        118,
        "LEVIR-CD-processed/train/A/train_118_6.png",
        86352,
        "a66176c0560acd2d6f211f1ecdb030e258d3b1fe897a194bf4219c4abb2f6864",
        "LEVIR-CD-processed/train/B/train_118_6.png",
        143966,
        "6d8a6d3b77d73ff2ef4784c4e75344fe2fdc074a81cdd8dda85d469e9ec793d0",
        "LEVIR-CD-processed/train/label/train_118_6.png",
        570,
        "6f86c353f121058e61f6c8808d550ed18980f50eca12ab9718d901f9b633c24d",
    ),
    (
        "train_122_19",
        "train",
        122,
        "LEVIR-CD-processed/train/A/train_122_19.png",
        178952,
        "5471c60cc49ac0d21b011da7879c1c6457b6172ca2177154e4778914d8665dcc",
        "LEVIR-CD-processed/train/B/train_122_19.png",
        143933,
        "c60e85ceb9374bdc83433d66a604dce96b36435776385703ebd58eb68eafce06",
        "LEVIR-CD-processed/train/label/train_122_19.png",
        2020,
        "b59ef303d0bbf55d379f92872549a9c416d2fe31b6ff7b86b3a14c473a1f2eed",
    ),
    (
        "train_148_23",
        "train",
        148,
        "LEVIR-CD-processed/train/A/train_148_23.png",
        155771,
        "3d24dbbfe0acb6202363d4c2b797e2011c0c5dc941c56b702db5cdabaabcb2aa",
        "LEVIR-CD-processed/train/B/train_148_23.png",
        110069,
        "f8da3c1034d006c5785e444cf267c067b70dff7279323cffd566c4c10986cc59",
        "LEVIR-CD-processed/train/label/train_148_23.png",
        1019,
        "238af4de4e48939613e2495c252e4f4be61234821b40a088ea0387e24d35640f",
    ),
    (
        "train_149_2",
        "train",
        149,
        "LEVIR-CD-processed/train/A/train_149_2.png",
        159393,
        "a8ca407697d2ff2ff1b984378bab37d9a8a38ed5b498ed6bc57b550031b62a61",
        "LEVIR-CD-processed/train/B/train_149_2.png",
        119432,
        "960e713e035f59cf4abf3be70a31c342639b586a305615a96a579b748e5d423e",
        "LEVIR-CD-processed/train/label/train_149_2.png",
        1264,
        "55a235d7ac65e349e6e0234db4b9773fbac4f4ba7254301c13d0301afaba4d68",
    ),
    (
        "train_226_21",
        "train",
        226,
        "LEVIR-CD-processed/train/A/train_226_21.png",
        93171,
        "7c53678d2e924cdbeed0dfebd2f68f20ad342d3435992a41e96c84dc29bd1048",
        "LEVIR-CD-processed/train/B/train_226_21.png",
        119881,
        "fd7a18a9fa5006b82ef0467ef5f9643c1c0e7cc37f3812fd227901af46b4a702",
        "LEVIR-CD-processed/train/label/train_226_21.png",
        1705,
        "6ebbd79eb77b132c0a426012041a9a75179eb1d67a2f9b134f5059a54a3b6cf6",
    ),
    (
        "train_231_15",
        "train",
        231,
        "LEVIR-CD-processed/train/A/train_231_15.png",
        137114,
        "ab6e2255b27f6c2c380f316d184eabb416962717926c457d7490ce76ff1af812",
        "LEVIR-CD-processed/train/B/train_231_15.png",
        139562,
        "d950d9cec25b4e311ac648273aced2a42026580e6dcd5e7d3095cd07164777f5",
        "LEVIR-CD-processed/train/label/train_231_15.png",
        2464,
        "ebbeb30aa17e981af374821305e3b1297d5b5b3dec55899cccfece5687932937",
    ),
    (
        "train_233_4",
        "train",
        233,
        "LEVIR-CD-processed/train/A/train_233_4.png",
        95164,
        "ee6f9a0a01d9e9471e8582a3a09c465529c5bda09c9e8a611410c75e07b67c8a",
        "LEVIR-CD-processed/train/B/train_233_4.png",
        121028,
        "74b03a8e84946aea267e1525fe5e996f3fd06cb4f57c2aebde231df947d276c4",
        "LEVIR-CD-processed/train/label/train_233_4.png",
        1910,
        "521350b380c7b16de629fe91be293e388240270e3a1f390c75c4ee9b6c740c06",
    ),
    (
        "train_237_19",
        "train",
        237,
        "LEVIR-CD-processed/train/A/train_237_19.png",
        76131,
        "b2efeeac24d86a998fe450c76b60188b62ad7a20777fb11d97ca5e82467946ba",
        "LEVIR-CD-processed/train/B/train_237_19.png",
        82074,
        "c0c66e152342aeda2f661f2eae7d243b89e69215b3ea1f36a6acd0abcccf30e5",
        "LEVIR-CD-processed/train/label/train_237_19.png",
        649,
        "482310846c93dbd09565b504ad74466b1729180eedb37c926e33e87c27908871",
    ),
    (
        "train_299_10",
        "train",
        299,
        "LEVIR-CD-processed/train/A/train_299_10.png",
        127875,
        "ada3288f9dba51648d5ef9cdab12c521d0f6ac8649b3940c3a3e6d06175abade",
        "LEVIR-CD-processed/train/B/train_299_10.png",
        112039,
        "214993a029573864fc440cc388178460789057c2a2f52c608fea9a1179af05c7",
        "LEVIR-CD-processed/train/label/train_299_10.png",
        868,
        "12251b46f7a468c530b15a9ee58a140df0e077e548bbdd58de443552aa7f52b1",
    ),
    (
        "train_300_4",
        "train",
        300,
        "LEVIR-CD-processed/train/A/train_300_4.png",
        124209,
        "7858c2f17ab438b7485d37bf44325d9b222602e30a1a32c1ad5bdcea19c213f8",
        "LEVIR-CD-processed/train/B/train_300_4.png",
        101030,
        "0dfa72b1b5082e16f31cad700b6b87ac58a07fa10efc023858aa42553c56446f",
        "LEVIR-CD-processed/train/label/train_300_4.png",
        713,
        "39e9f31e0f54d1b8ca347740cd0e9c81a4fb415a47c3495ae97cfe38f8d12ec4",
    ),
    (
        "train_305_9",
        "train",
        305,
        "LEVIR-CD-processed/train/A/train_305_9.png",
        128725,
        "0253340459757b3f8839d01a59fda34898fcd6a7f83be181533a7dc1806ced1a",
        "LEVIR-CD-processed/train/B/train_305_9.png",
        132218,
        "f8977e86770c9eb2d698959a8023e41069bd8162ac857d969ddfe687b83e9a44",
        "LEVIR-CD-processed/train/label/train_305_9.png",
        2138,
        "4f587efb362137e989ab9b548a3b4f6a8d555cecf0c13ac997b78ab948524419",
    ),
    (
        "train_311_6",
        "train",
        311,
        "LEVIR-CD-processed/train/A/train_311_6.png",
        108147,
        "a9b1cdaee5b1670c82b30cbc638688490af5b0890fea3689afbe6c0bd0cce686",
        "LEVIR-CD-processed/train/B/train_311_6.png",
        107827,
        "10839b3eb6984a6a4592d5e19eca7dc24fe4669c38e5795a28524bcffda4d423",
        "LEVIR-CD-processed/train/label/train_311_6.png",
        523,
        "a04a7b82bbff4e85690dec9f5dc87b2ecaf35335fdb6e2f2a7849b0ac287bfb7",
    ),
    (
        "train_314_20",
        "train",
        314,
        "LEVIR-CD-processed/train/A/train_314_20.png",
        164138,
        "f9ac244f80c5259be068839a65c6981fd12afe46ea810eb46db0b6b180139ba0",
        "LEVIR-CD-processed/train/B/train_314_20.png",
        115166,
        "fb73a00e8f5eb9fb2a2bc92315dad913bb0ec4306fa86493873bee8c4ca8b276",
        "LEVIR-CD-processed/train/label/train_314_20.png",
        621,
        "60d82dd8446f0299e6cf64bf541a6fb8f050e6f35ffffe5896e829842edd2b91",
    ),
    (
        "train_346_1",
        "train",
        346,
        "LEVIR-CD-processed/train/A/train_346_1.png",
        146271,
        "4701c8519c48267ea06c61bee3b88a66f15ffd2c1b9f5e6a15a82b8d82e0b032",
        "LEVIR-CD-processed/train/B/train_346_1.png",
        129026,
        "1675df817f982f90ce0aecb73198c284e4ce567c7b438ca0bb6b70d71deaebcf",
        "LEVIR-CD-processed/train/label/train_346_1.png",
        725,
        "f5048d3f0a2adf8ce227a684efed7173ef1006ead78d0737fa920d1f14c49d3d",
    ),
    (
        "train_358_1",
        "train",
        358,
        "LEVIR-CD-processed/train/A/train_358_1.png",
        125015,
        "eaef39114aa8c025fbba056b3848ce50c60593e78f7ba471b8d3e0d7f5d9dffc",
        "LEVIR-CD-processed/train/B/train_358_1.png",
        112562,
        "ae2a21003846ce8b85b0ee629cc9195e4377ae87dc571b5a7e1cd829fc9d9148",
        "LEVIR-CD-processed/train/label/train_358_1.png",
        621,
        "498292479419bb2bb4147cb7ca063d3f4011fb251d3621e3dc7f7b180525cfc7",
    ),
    (
        "train_359_0",
        "train",
        359,
        "LEVIR-CD-processed/train/A/train_359_0.png",
        154413,
        "36bf727867420dd1f6237f8ee30833a46d5e660c4f8ce833eed9436e47be978e",
        "LEVIR-CD-processed/train/B/train_359_0.png",
        113445,
        "2f6a90370b6c1ccc8e34f640cd825a2e741e91b1be378b0d8a87be0af5bb2957",
        "LEVIR-CD-processed/train/label/train_359_0.png",
        289,
        "b8c8f0fa51bc69966c5dd816abd90585053d73764a9da18adc27924906f00c0e",
    ),
    (
        "train_362_9",
        "train",
        362,
        "LEVIR-CD-processed/train/A/train_362_9.png",
        96493,
        "60c3c4a62a54db3f0ba36c3da583ef51c6955bd061ababcd064a53f78e94b151",
        "LEVIR-CD-processed/train/B/train_362_9.png",
        112987,
        "1803f612476b41ef178493a239217ce27cc587e3a3bd4fc2ed1a5d7815d92814",
        "LEVIR-CD-processed/train/label/train_362_9.png",
        558,
        "557645e9b88e79824e1a181635d4e959c791f4e854545f2b7b0e1902b515e5f5",
    ),
    (
        "train_404_13",
        "train",
        404,
        "LEVIR-CD-processed/train/A/train_404_13.png",
        120659,
        "0e0a838ed3a260cb8be581094bc2a47e3497f54cc132ec108c7b97acab7224d9",
        "LEVIR-CD-processed/train/B/train_404_13.png",
        121388,
        "9bbd3f535634e35eb334dbabe9e3de2a985df19f1d894252bcf4cc8d51b5e4d9",
        "LEVIR-CD-processed/train/label/train_404_13.png",
        785,
        "b940c7e24ebd272a01c3f903ebc779f7865d531d9f2335bd2090d4d80398c993",
    ),
    (
        "train_414_10",
        "train",
        414,
        "LEVIR-CD-processed/train/A/train_414_10.png",
        90392,
        "0735afe057b19a64e9f8d4da460595f63377ebcf8c2ecbc02912c8f5901b5e58",
        "LEVIR-CD-processed/train/B/train_414_10.png",
        130745,
        "7ab753f944389aac65b5623fd89d40310c97da9ad68ccfa3960bdcc839020dfc",
        "LEVIR-CD-processed/train/label/train_414_10.png",
        1848,
        "cbbbf236d8b9a00c4d539b0b85ea20d2f904a56255f7c7e5a9edcd0eaf9308e2",
    ),
    (
        "train_425_9",
        "train",
        425,
        "LEVIR-CD-processed/train/A/train_425_9.png",
        95801,
        "e7c972f7bae56775ef7b20e93e5bc655bca63b22ebf5f1976a74093d9ff3bafb",
        "LEVIR-CD-processed/train/B/train_425_9.png",
        124006,
        "9f752a6b3f02e4220008f102ba044eb47fccd6692d3dc8cb7f69cb5bf28f6b81",
        "LEVIR-CD-processed/train/label/train_425_9.png",
        1833,
        "229cca6a374efaf812f3041820927d2c306b51b51768a426fd568d2fd3baf8d8",
    ),
    (
        "train_442_20",
        "train",
        442,
        "LEVIR-CD-processed/train/A/train_442_20.png",
        67918,
        "4af3eb2d979ba8b85241aced2265e95f39c25bc2b00c43b7e1cb771e38a524cb",
        "LEVIR-CD-processed/train/B/train_442_20.png",
        102805,
        "1d773004a5b13ee8f261fa23d85320a0339b1114c26e78fd4b8088defe108750",
        "LEVIR-CD-processed/train/label/train_442_20.png",
        1124,
        "3e503e368f3595f99b8d992a514e0e872856d5a995ccebeaf7983080e6945d65",
    ),
    (
        "val_6_24",
        "validation",
        6,
        "LEVIR-CD-processed/val/A/val_6_24.png",
        112061,
        "c99601cb38f4a4590d2ba7f00867a6d10390c9fbcce73f98233290c1af6d0abc",
        "LEVIR-CD-processed/val/B/val_6_24.png",
        145826,
        "eae0b0a56004df2d16ab77454ce123c515bf24fb1ddc38f4c818361c2f3be660",
        "LEVIR-CD-processed/val/label/val_6_24.png",
        2970,
        "4de2c752eb8ffefc0badd13e071b6de40d1418db06ef416859265d2f5398632a",
    ),
    (
        "val_13_9",
        "validation",
        13,
        "LEVIR-CD-processed/val/A/val_13_9.png",
        164058,
        "720c407b81695eb1b09163cddb1e0ce13165f38d0bf3e643d8f081a23034350e",
        "LEVIR-CD-processed/val/B/val_13_9.png",
        149631,
        "0e76fabab28ea263324d00d9eda95371cc9d39c47d3047e96aeabea26fcbb991",
        "LEVIR-CD-processed/val/label/val_13_9.png",
        896,
        "d85cab576ca4a8178092c41aa4927adc3bc6c81c00b258ad2da5932e07ad729e",
    ),
    (
        "val_34_5",
        "validation",
        34,
        "LEVIR-CD-processed/val/A/val_34_5.png",
        68368,
        "d4a8849b5110bd5a9efeeeb6f472279c018355809e19be3d2c47a62958eec24d",
        "LEVIR-CD-processed/val/B/val_34_5.png",
        140766,
        "c0aeafe424e614fdcefa84cc46a5f5731cc70204f34596a3dcd1d87e0cd68332",
        "LEVIR-CD-processed/val/label/val_34_5.png",
        3085,
        "bf0bbc96735ab9f8462ad0c0ba2eb1e36a8489f06601f2476bfb927e27ee54af",
    ),
    (
        "val_37_9",
        "validation",
        37,
        "LEVIR-CD-processed/val/A/val_37_9.png",
        162744,
        "d404664e468e74398f9c0088154e99dda8e7998f33107e5e7cf3eeb1692f90d5",
        "LEVIR-CD-processed/val/B/val_37_9.png",
        121061,
        "b959c466771b15ce39469e2a4645830f3d1775b715ac63b71cfd51010bf52769",
        "LEVIR-CD-processed/val/label/val_37_9.png",
        519,
        "e8b6ab1a26a571341589a869890bc43c9d0cbd0fea7467ed0c9df002347b9bad",
    ),
    (
        "val_39_16",
        "validation",
        39,
        "LEVIR-CD-processed/val/A/val_39_16.png",
        136529,
        "0b0fc77a3afd48341c2646518046fa6e87c75eff1a8ef464ccbe7f2bab273995",
        "LEVIR-CD-processed/val/B/val_39_16.png",
        132874,
        "225cd0cbbfc5e653be61ad332ff8578b4ac2bdc187d3cb2e5a2ffca928c69648",
        "LEVIR-CD-processed/val/label/val_39_16.png",
        1498,
        "ac32e38e175ab43aa31b318083e56203e52c7c0124f4378bc858543d69973f4d",
    ),
    (
        "val_40_15",
        "validation",
        40,
        "LEVIR-CD-processed/val/A/val_40_15.png",
        136860,
        "fef12f294e9f408371f64fa205766bc08ac9ee8187178834f2312c8b59b36816",
        "LEVIR-CD-processed/val/B/val_40_15.png",
        125469,
        "d30ce4e1e8a6350ef61bb6af8de0455f26985d6d135bcbf782acc435880ad08f",
        "LEVIR-CD-processed/val/label/val_40_15.png",
        1086,
        "19163c01367aa7b1caa2952ccd946fea1f9107c61169a5ba5a57ed11f1c063ee",
    ),
    (
        "val_42_5",
        "validation",
        42,
        "LEVIR-CD-processed/val/A/val_42_5.png",
        141920,
        "ce1807e780ad6a04b711f80ef96c8d6ae99de39c4d8ba51d69048a2bccc5d106",
        "LEVIR-CD-processed/val/B/val_42_5.png",
        103066,
        "440153679cf5159a70879c4a602decb49e9d51a0678a3bfb9940acfa295d4048",
        "LEVIR-CD-processed/val/label/val_42_5.png",
        513,
        "e84ba0e900478d275cf8d4ccbb18679c2193a7aa99df7f0cb499a1ccdcde681a",
    ),
    (
        "val_43_16",
        "validation",
        43,
        "LEVIR-CD-processed/val/A/val_43_16.png",
        131135,
        "5ff6bac90b14c45c467ad2bbaaaeedf47623d8aa516eb55079b6bea9b2e47ce5",
        "LEVIR-CD-processed/val/B/val_43_16.png",
        115808,
        "1ca82db4bc324eade98c0f0f337ef6c0e5efc59ee5b8a50e6f955277b16a8352",
        "LEVIR-CD-processed/val/label/val_43_16.png",
        1575,
        "15b5e948b2d085887afb9d26faec15f509ce66859241965cc74ad8c99afc5160",
    ),
    (
        "test_2_20",
        "test",
        2,
        "LEVIR-CD-processed/test/A/test_2_20.png",
        141865,
        "f7e0424b4b3b8dc7c2b0c3eb16cc3493b5676360314656c01014709fc744c4ce",
        "LEVIR-CD-processed/test/B/test_2_20.png",
        131326,
        "4cc3a90d556c66a92e3a8750b7e52523b0001cdf7d6c34ecdb9104d1053121ff",
        "LEVIR-CD-processed/test/label/test_2_20.png",
        671,
        "d8dab0716aed9fa32d3b39a8af5e955a96909709ad9dfc7feba576322e064a9e",
    ),
    (
        "test_3_19",
        "test",
        3,
        "LEVIR-CD-processed/test/A/test_3_19.png",
        145215,
        "cb4dc060d4cbd5d10decd84d0eb03b94f533cf59dba47f45f59eeee73eec57fe",
        "LEVIR-CD-processed/test/B/test_3_19.png",
        141296,
        "456e0b0900414f703720b9eb9b341423e368e7fbd8f57ede0491f171d6b29640",
        "LEVIR-CD-processed/test/label/test_3_19.png",
        654,
        "cfeb5522eb0530f56ee3b172d7d5597355e7f9ec7691bc0b76dd4251ca2ab21e",
    ),
    (
        "test_4_19",
        "test",
        4,
        "LEVIR-CD-processed/test/A/test_4_19.png",
        137794,
        "11bbfbe833c59c003b2833b6d1a8e7569e26a6b19d4b8681c157df0cc83c75fe",
        "LEVIR-CD-processed/test/B/test_4_19.png",
        122359,
        "358fa76346fc13ce816cf21e52171eae98bcec0abbfd882611fb6fb6b695bd84",
        "LEVIR-CD-processed/test/label/test_4_19.png",
        696,
        "4f624a332fc2533f2b512887ffb117ebe2a9cc87d1abb830bca1fde448f82810",
    ),
    (
        "test_8_14",
        "test",
        8,
        "LEVIR-CD-processed/test/A/test_8_14.png",
        153854,
        "7d07053d258dfadaf70e855b31d2cb63c633a88e9bf7907763c3fc3d4dd81f04",
        "LEVIR-CD-processed/test/B/test_8_14.png",
        135059,
        "2cfca66e9eaf33f3364efefb95902a6bd8b431db8cfedc75ef7c62578c4b3ffa",
        "LEVIR-CD-processed/test/label/test_8_14.png",
        2137,
        "c96a15c5c6c61659aa6fb10ed5f6fed50a7c4aea309e5a96da51efb4315115ee",
    ),
    (
        "test_9_2",
        "test",
        9,
        "LEVIR-CD-processed/test/A/test_9_2.png",
        135724,
        "7451cb17c2c51c1f9aeeb4d47be036a5a1dab760f983f2419986543697ed3b6f",
        "LEVIR-CD-processed/test/B/test_9_2.png",
        136605,
        "0b88777bfe60ad1fb68716102830b45c9408b351592b0455399fe5734c026808",
        "LEVIR-CD-processed/test/label/test_9_2.png",
        625,
        "2f84f033c2297ee17a29a84309672e1670f085231b0479a7c2c670f9057865ed",
    ),
    (
        "test_10_8",
        "test",
        10,
        "LEVIR-CD-processed/test/A/test_10_8.png",
        93161,
        "5b82e3fd3c35beae8b24a9c3934d68b4c1439dcc1c5ada90b1f0592b514e1068",
        "LEVIR-CD-processed/test/B/test_10_8.png",
        141767,
        "2403fdc75bfe267ee0f369229c943cfbe4da8f58416c1bf4eff56e3069816431",
        "LEVIR-CD-processed/test/label/test_10_8.png",
        1988,
        "e518a9651321ba95036fc8b0d7151bdd5e93f624059561a418633b0eedda4791",
    ),
    (
        "test_20_5",
        "test",
        20,
        "LEVIR-CD-processed/test/A/test_20_5.png",
        150549,
        "6c1eaa4030376693128db5f8cdf215308853b83ef7dbb5cfcd54f414b2955578",
        "LEVIR-CD-processed/test/B/test_20_5.png",
        146915,
        "452b474418576900a9d9d36b20107c8ee6cdf9fc2f713ca918eee66c05db8979",
        "LEVIR-CD-processed/test/label/test_20_5.png",
        2399,
        "19768846d74a9f8754d6d00d4257bbaf5e62e319002fe65bb4430c98029bd545",
    ),
    (
        "test_23_1",
        "test",
        23,
        "LEVIR-CD-processed/test/A/test_23_1.png",
        165292,
        "c60bbc22f9998c825dfc6db05fa34fb05edbe118308ffb257d0d20e9c2330daf",
        "LEVIR-CD-processed/test/B/test_23_1.png",
        141676,
        "4cba47cdcd533a59d412a98c9c6b1910cced3c94b670b822caeaefa03ff8d4d1",
        "LEVIR-CD-processed/test/label/test_23_1.png",
        1288,
        "7643b44c259c6e06434f683a5b664c68e34d2f6c2468f52593eb2033faf58223",
    ),
    (
        "test_28_2",
        "test",
        28,
        "LEVIR-CD-processed/test/A/test_28_2.png",
        164834,
        "67d73603c0704a331f377513eeca9f508f66cf0490c6df22a1596bccd563af20",
        "LEVIR-CD-processed/test/B/test_28_2.png",
        146091,
        "e4edadee584b74c0aaddc38be2d7805681efc2770b4d91896f1be2a38bf7c1da",
        "LEVIR-CD-processed/test/label/test_28_2.png",
        2182,
        "bfd31504e625c777f7da022b47f3c0b4129d089a98e2539b59f03d71fc430b2b",
    ),
    (
        "test_39_13",
        "test",
        39,
        "LEVIR-CD-processed/test/A/test_39_13.png",
        147482,
        "1638b941958e84e6d842b83129f7dfbec7c55bd6969f4ad99f39367ed1fa0e41",
        "LEVIR-CD-processed/test/B/test_39_13.png",
        110132,
        "5e023b21e986a79f996fd99f9ced0b3a7b680cf38645b95b0dd7e1fa5f9a9701",
        "LEVIR-CD-processed/test/label/test_39_13.png",
        1546,
        "44ac4b77080946a989b8c614adafc8366508f3776838ac8e175d543638e4619f",
    ),
    (
        "test_40_5",
        "test",
        40,
        "LEVIR-CD-processed/test/A/test_40_5.png",
        132757,
        "0378c28073526d476b6b5f4ca6d5e2f0d36d0e97feb787e531859389aa77d4fe",
        "LEVIR-CD-processed/test/B/test_40_5.png",
        133560,
        "a07fd9e242edd035d007e878e23211985c8068515885dc59357b4b98d11a15ab",
        "LEVIR-CD-processed/test/label/test_40_5.png",
        1857,
        "4515243885a055f616254a4fcbb58e6a5b783e1a42e8f7dbdaabcd2f1e151013",
    ),
    (
        "test_45_10",
        "test",
        45,
        "LEVIR-CD-processed/test/A/test_45_10.png",
        153525,
        "c7d7ab34df06a94b6dda3c5f37e209736c79fc567b777276c39770352de3ec4e",
        "LEVIR-CD-processed/test/B/test_45_10.png",
        138345,
        "cf772c4ed819ca5b56f3cf0ce16c1da89dffa892a47b2aed1f71f78405524da8",
        "LEVIR-CD-processed/test/label/test_45_10.png",
        3240,
        "2ebbeb40a0ad4a58ad537b73f244c301ff9285abd8f191cb31f53ba28d88edf6",
    ),
    (
        "test_46_11",
        "test",
        46,
        "LEVIR-CD-processed/test/A/test_46_11.png",
        157443,
        "9c554db0ca372ab0a7735fbe5b19960bbc9efbcb54bfe0c06fc341b8c8a20863",
        "LEVIR-CD-processed/test/B/test_46_11.png",
        144440,
        "9ef7c509d70ed8e12e740694a2333df7a1c2909d66c93c8e7227f2dcd76466d7",
        "LEVIR-CD-processed/test/label/test_46_11.png",
        3139,
        "c8817bd021f7f701a7eda3509ce05ac41d5a897804cda2ae3305436308314ffb",
    ),
    (
        "test_53_9",
        "test",
        53,
        "LEVIR-CD-processed/test/A/test_53_9.png",
        140640,
        "579903243f14d8473101dcdc0aef1bc6890bbf877eb039e63874f103b900c9d3",
        "LEVIR-CD-processed/test/B/test_53_9.png",
        145631,
        "f63d62f6c2b15f6a91d866947c4eeec77734ca8e6eb35d8c41646282cbb7265b",
        "LEVIR-CD-processed/test/label/test_53_9.png",
        844,
        "ea1a364a59d658523373e0607fdbdc7cff2ac05b9a66bf53e105e4f25eeb9a26",
    ),
    (
        "test_54_6",
        "test",
        54,
        "LEVIR-CD-processed/test/A/test_54_6.png",
        101916,
        "c8daf01f0b29fd90997fb41c51a7702b58dce8773165e460466f265b6ab50be3",
        "LEVIR-CD-processed/test/B/test_54_6.png",
        134302,
        "b4c8d761ddf5a2106397ce33e97dc45db10946495a9faabfeab27ac4c9f17255",
        "LEVIR-CD-processed/test/label/test_54_6.png",
        1206,
        "486c67f0b45037d52b6ae07c4eafe812dd5aff9c8c3e7dd0ddb404df4f38e0e0",
    ),
    (
        "test_72_15",
        "test",
        72,
        "LEVIR-CD-processed/test/A/test_72_15.png",
        74043,
        "50f30d1789bea1662e127b0ca69d93d4a4ce425d4142aa43fe7f88797a3888f3",
        "LEVIR-CD-processed/test/B/test_72_15.png",
        130279,
        "d56981df93439ff11721f8496d653762bcb66c30ba86cbc0e130222902794ead",
        "LEVIR-CD-processed/test/label/test_72_15.png",
        2579,
        "806ece0e09614920b0e98366e732b2bf1b332fb62494d22f98407a551d1d1231",
    ),
    (
        "test_74_0",
        "test",
        74,
        "LEVIR-CD-processed/test/A/test_74_0.png",
        124484,
        "4da5be567461bc44eab5f6aa2fbd6949140bf8eb04d8effd056d17b1c82a0772",
        "LEVIR-CD-processed/test/B/test_74_0.png",
        113788,
        "9484c2307f94c8c7f90f96f294dd97470a8b01ab10398a465305131825dedd94",
        "LEVIR-CD-processed/test/label/test_74_0.png",
        941,
        "ea74105f94f99ed51d602e138bc5d684bd8be792cacda39fe144af3b537b1a2f",
    ),
    (
        "test_76_22",
        "test",
        76,
        "LEVIR-CD-processed/test/A/test_76_22.png",
        172511,
        "1be2a8e4ba0c78975af8a3a0034d682823fa2bbf7a3fc4f5232377b851d12efc",
        "LEVIR-CD-processed/test/B/test_76_22.png",
        137754,
        "d8e950323a3b442371b91ab34cc446dfde240839959cee69c94c6db1ee5cad58",
        "LEVIR-CD-processed/test/label/test_76_22.png",
        466,
        "7dd0477543df99f94273ad512af901349face92c61a16dc0e4fd90dec2c3d856",
    ),
    (
        "test_79_15",
        "test",
        79,
        "LEVIR-CD-processed/test/A/test_79_15.png",
        161096,
        "adea19ea0168f518e79936e8ff1eaed05146bad80a29b34d7bc930816f2b7fb9",
        "LEVIR-CD-processed/test/B/test_79_15.png",
        124633,
        "fc8d0e572c8229cc2bd7d251666c6cda4e3f09aac7d0c09cbdb56ea2e1127f6a",
        "LEVIR-CD-processed/test/label/test_79_15.png",
        394,
        "5231f329ebcec7e74d07712eb81a5e6bc433835534a2a0a3beda8a15a2f04451",
    ),
    (
        "test_100_24",
        "test",
        100,
        "LEVIR-CD-processed/test/A/test_100_24.png",
        145607,
        "8b5642d499d8d97d62b19ca8e4b3aec33d550df451a4bccf8cca39f8dd80fd52",
        "LEVIR-CD-processed/test/B/test_100_24.png",
        140629,
        "fa8fa44dc7da0179d91b36aa406ee2285f75094e90835bfaa9cfff4674bd1e79",
        "LEVIR-CD-processed/test/label/test_100_24.png",
        2854,
        "1a37972c68b2803be80685c45710466fddbc6c78802c64bc2b33d64690549ed5",
    ),
    (
        "test_101_22",
        "test",
        101,
        "LEVIR-CD-processed/test/A/test_101_22.png",
        110280,
        "f5389f31868b635af8291a6fbeea17e4f31c2b6bfa86d976e33d5ff9c55b6221",
        "LEVIR-CD-processed/test/B/test_101_22.png",
        141281,
        "5c4adc15a1af7b4319ccd81be827d4302c2756c69485603f9e7faf8835bfe7b7",
        "LEVIR-CD-processed/test/label/test_101_22.png",
        854,
        "885b7450cc6ac965b6a9a541cfd525e8e66dc9f9eb5b810333ad9a1e6e87b45e",
    ),
    (
        "test_115_3",
        "test",
        115,
        "LEVIR-CD-processed/test/A/test_115_3.png",
        154916,
        "e7328316dc10f2f2a592e7d1a0f8c767318fc245544d0e367c4f48b408fe9e2a",
        "LEVIR-CD-processed/test/B/test_115_3.png",
        125364,
        "11e124e2b9369bb7b63464cd17b22d3b8e908ba80bddfeaa4cd587c64fcf7d9c",
        "LEVIR-CD-processed/test/label/test_115_3.png",
        659,
        "71412fc144315b89aa4777a034fdc53d5c35e0a0d27a29872ef48c298e26548d",
    ),
    (
        "test_118_3",
        "test",
        118,
        "LEVIR-CD-processed/test/A/test_118_3.png",
        169357,
        "db3cd0568118dde3ac7a92161d07813807249bebc387b290d408413f3fc9042b",
        "LEVIR-CD-processed/test/B/test_118_3.png",
        141099,
        "78beb206e121989f50182899f29e392efb9b3ade8797d4fde3f2207120a6d6b7",
        "LEVIR-CD-processed/test/label/test_118_3.png",
        2413,
        "74dc6d03c847b7d9c4ff1a1d000070f6d4ccd8294cbaa001dd66c4fd25f2388e",
    ),
    (
        "test_119_21",
        "test",
        119,
        "LEVIR-CD-processed/test/A/test_119_21.png",
        136388,
        "db596e11eb93487868b0189b718126f5c4989e069917f1756e474a9c030629f2",
        "LEVIR-CD-processed/test/B/test_119_21.png",
        111422,
        "fa61de13dc121afafe80c259a952c10c2cabeb89b44285ec97f01004d119063c",
        "LEVIR-CD-processed/test/label/test_119_21.png",
        387,
        "28f4691d040d849dca4b7b70c9251f7faa07ffe0244c1431edda5512d225569c",
    ),
)

SAMPLE_LABEL_SOURCE = f"{CORPUS_NAME}; {CORPUS_RELEASE}; {CORPUS_LICENSE}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_members() -> dict[str, tuple[int, str]]:
    out = {}
    for record in SAMPLE_RECORDS:
        for offset in (3, 6, 9):
            out[record[offset]] = (record[offset + 1], record[offset + 2])
    return out


def _hub_download_tarball(destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(DATASET_ID, TAR_NAME, repo_type="dataset", revision=DATASET_REVISION, local_dir=str(destination.parent))


def fetch_tarball(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> Path:
    """The pinned dataset tarball in the cache, fetched from the Hub at the immutable revision when absent, and
    refused on a size or SHA-256 mismatch (the 3.8 GB file is hashed once per call)."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / TAR_NAME
    if not local.is_file() or local.stat().st_size != TAR_BYTES:
        if fetcher is not None:
            local.write_bytes(fetcher(CORPUS_BASE_URL + TAR_NAME))
        else:
            _hub_download_tarball(local)
    size = local.stat().st_size
    digest = _sha256_file(local)
    if size != TAR_BYTES or digest != TAR_SHA256:
        raise ValueError(f"{TAR_NAME}: {size} bytes with sha256 {digest[:16]}…, pinned {TAR_BYTES} / {TAR_SHA256[:16]}…")
    return local


def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def extract_pinned_members(tar_path: str | Path, *, cache_dir: str | Path | None = None) -> dict[str, bytes]:
    """Stream through the tarball once and copy out exactly the pinned members (no `extractall`, no paths from
    the archive: each is written under `<split>_<folder>_<name>` in `cache_dir/crops/`), refusing a size or
    digest mismatch."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    crops = cache / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    wanted = _pinned_members()
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            name = _member_name(member.name)
            if name not in wanted or not member.isfile():
                continue
            size, sha = wanted[name]
            handle = archive.extractfile(member)
            data = handle.read() if handle is not None else b""
            if len(data) != size or _sha256_bytes(data) != sha:
                raise ValueError(
                    f"{name}: {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, pinned {size} / {sha[:16]}…"
                )
            (crops / _cache_name(name)).write_bytes(data)
            out[name] = data
            if len(out) == len(wanted):
                break
    missing = sorted(set(wanted) - set(out))
    if missing:
        raise ValueError(f"tarball does not contain {len(missing)} pinned members, e.g. {missing[:3]}")
    return out


def _cache_name(member: str) -> str:
    """`LEVIR-CD-processed/test/A/test_1_2.png` -> `test_A_test_1_2.png` (flat, no archive paths)."""
    parts = member.split("/")
    return "_".join(parts[-3:])


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, bytes]]:
    """Every pinned crop's before / after / label bytes, keyed by crop key: from the extracted cache when every
    file is present with its pinned digest, otherwise from the (verified) tarball."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    crops = cache / "crops"
    wanted = _pinned_members()
    cached: dict[str, bytes] = {}
    for member, (size, sha) in wanted.items():
        local = crops / _cache_name(member)
        if local.is_file() and local.stat().st_size == size:
            data = local.read_bytes()
            if _sha256_bytes(data) == sha:
                cached[member] = data
    if len(cached) != len(wanted):
        cached = extract_pinned_members(fetch_tarball(cache_dir=cache, fetcher=fetcher), cache_dir=cache)
    return {r[0]: {"before": cached[r[3]], "after": cached[r[6]], "label": cached[r[9]]} for r in SAMPLE_RECORDS}


def read_corpus(files: Mapping[str, Mapping[str, bytes]]) -> dict[str, list[dict[str, Any]]]:
    """Decode the verified bytes into `{id, before, after, label}` records grouped by role (train / validation /
    test)."""
    import tempfile

    splits: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for key, role, pair, before_member, *_rest in SAMPLE_RECORDS:
        if key not in files:
            raise ValueError(f"corpus is missing {key}")
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for part in ("before", "after", "label"):
                paths[part] = Path(tmp) / f"{part}.png"
                paths[part].write_bytes(files[key][part])
            before = read_image(paths["before"])
            after = read_image(paths["after"])
            label = read_mask(paths["label"])
        raw = {
            "id": f"{role}-{len(splits[role]):03d}",
            "source_id": key,
            "region": f"{role}-pair-{pair}",  # the 1024 × 1024 source pair the crop was cut from
            "split": role,
            "before": before,
            "after": after,
            "label": label,
            "source": f"{CORPUS_BASE_URL}{TAR_NAME}#{before_member}",
        }
        splits[role].append(check_record(raw))
    return splits


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus (roles = the dataset's own splits)."""
    return read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no pair (by pixel digest) and no source pair (by `region`) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    regions: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = pair_digest(record)
            if key in seen and seen[key] != name:
                raise ValueError(f"pair {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
            region = record.get("region")
            if region:
                if region in regions and regions[region] != name:
                    raise ValueError(f"source pair {region!r} has crops in both {regions[region]} and {name}")
                regions[region] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train / validation / test after de-duplicating pairs. Crops of one
    scene are near-duplicates; group them yourself (one scene per split) when that matters."""
    import random

    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = pair_digest(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(f"split leaves {len(splits['train'])} training pairs; at least {MIN_RECORDS} are required")
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, before, after, label}` records from a directory or a zip holding `pairs.csv` (columns `id`,
    `before`, `after`, `label`) beside same-sized RGB PNG / JPEG images and 0 / 255 label PNGs; files are decoded
    from bytes, never extracted to disk."""
    import tempfile

    source = Path(path)
    if source.is_dir():
        table = (source / "pairs.csv").read_text(encoding="utf-8")
        loader = lambda name: (source / name).read_bytes()  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "pairs.csv" not in members:
            raise ValueError("BYOD zip must contain pairs.csv")
        table = archive.read(members["pairs.csv"]).decode("utf-8")
        loader = lambda name: archive.read(members[name])  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding pairs.csv and the image files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "before", "after", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"pairs.csv is missing columns {sorted(missing)}")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            record: dict[str, Any] = {"id": row["id"]}
            for part in ("before", "after"):
                image_path = Path(tmp) / f"{part}{Path(row[part]).suffix.lower() or '.png'}"
                image_path.write_bytes(loader(row[part]))
                record[part] = read_image(image_path)
            if row.get("label"):
                label_path = Path(tmp) / "label.png"
                label_path.write_bytes(loader(row["label"]))
                record["label"] = read_mask(label_path)
            out.append(record)
    return out


def write_sample_pair(
    record: Mapping[str, Any], before_path: str | Path, after_path: str | Path, label_path: str | Path
) -> dict[str, str]:
    """Write one record as two RGB PNGs and a 0 / 255 label PNG (the BYOD shape) and return the paths."""
    import numpy as np
    from PIL import Image

    paths = {"before": Path(before_path), "after": Path(after_path), "label": Path(label_path)}
    paths["before"].parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(record["before"], dtype=np.uint8)).save(paths["before"])
    Image.fromarray(np.asarray(record["after"], dtype=np.uint8)).save(paths["after"])
    label = np.asarray(record["label"], dtype=np.int64)
    Image.fromarray(np.where(label == 1, 255, 0).astype(np.uint8)).save(paths["label"])
    return {k: str(v) for k, v in paths.items()}


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the pairs table of a split (id, before, after, label, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "before", "after", "label", "region", "source"])
        writer.writeheader()
        for record in records:
            stem = record.get("source_id", record["id"])
            writer.writerow(
                {
                    "id": record["id"],
                    "before": f"{stem}_A.png",
                    "after": f"{stem}_B.png",
                    "label": f"{stem}_label.png",
                    "region": record.get("region", ""),
                    "source": record.get("source", ""),
                }
            )
    return out


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate every split and summarise the dataset (counts, change balance, digests) for provenance exports."""
    summary: dict[str, Any] = {"model_id": MODEL_ID, "sample_size": SAMPLE_SIZE, "splits": {}}
    for name, records in splits.items():
        report = validate_dataset(records, min_records=1)
        summary["splits"][name] = {
            "n_records": report["n_records"],
            "sizes": report["sizes"],
            "change_fraction": report["change_fraction"],
            "ignored_pixels": report["ignored_pixels"],
            "regions": sorted({str(r.get("region", "")) for r in records if r.get("region")}),
            "digest": report["digest"],
        }
    summary["disjoint"] = check_split_disjoint(splits)
    digests = json.dumps({k: v["digest"] for k, v in summary["splits"].items()}, sort_keys=True)
    summary["digest"] = hashlib.sha256(digests.encode("utf-8")).hexdigest()
    return summary

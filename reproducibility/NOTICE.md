# Sources, attribution and terms

The first-party software contributions listed in `SOFTWARE_LICENSE_FILES.json` use Apache-2.0; see `LICENSE` and `LICENSE_SCOPE.md`. This software grant does not cover the research observations, fitted states, pixel fields, photographs, annotations, image derivatives, model weights, fonts or third-party material. No blanket data license is supplied for the original numerical contributions.

## Upstream software

DINOv2, AnomalyDINO, PatchCore and SubspaceAD retain their Apache-2.0 source terms; MH-PatchCore retains MIT. DINOv2's nested CLIP code retains its MIT notice. Original author names, copyright headers and notices remain with the corresponding sources. In particular, the MH-PatchCore MIT and CLIP MIT names and the Amazon PatchCore notice are upstream legal attribution, not the authorship of this paper.

The retained license files are:

- `third_party/DINOv2/LICENSE`
- `third_party/DINOv2/dinov2/thirdparty/CLIP/LICENSE`
- `third_party/AnomalyDINO/LICENSE`
- `third_party/PatchCore/LICENSE`
- `third_party/PatchCore/NOTICE`
- `third_party/MH_PatchCore/LICENSE`
- `third_party/SubspaceAD/LICENSE`

The copied backbone wrapper `code/directional_adapter/upstream_backbones_source.py` originates from pinned AnomalyDINO and retains its source attribution. Source commits and checkpoint identities are listed in `docs/DATA_SOURCES.md`. A code license does not settle the terms of every training image, checkpoint or derived data object.

## Dataset objects

VisA photographs and GT use the provider's [CC BY 4.0 dataset terms](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET). Preserve the dataset citation, source identity and description of crops, resizing and overlays when reusing selected objects.

Selected MANTA photographs and source annotations are credited to **Lei Fan, Dongdong Fan, Zhiguang Hu, Yiwen Ding, Donglin Di, Kai Yi, Maurice Pagnucco and Yang Song**, authors of MANTA (CVPR 2025). The deposits used are:

- [Tiny 27934938.v3](https://doi.org/10.6084/m9.figshare.27934938.v3)
- [Native paddy 27938454.v1](https://doi.org/10.6084/m9.figshare.27938454.v1)
- [Native maize 27935718.v1](https://doi.org/10.6084/m9.figshare.27935718.v1)
- [Native square button cap 27937521.v1](https://doi.org/10.6084/m9.figshare.27937521.v1)

These exact source versions identify [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The selected MANTA photographs and source annotations identified in `docs/MANTA_MATERIALS.csv` are reproduced and distributed under the CC BY 4.0 license attached to these exact Figshare versions. That license permits reproduction, adaptation and redistribution with attribution, a license link and identification of changes, including the crops, resizing and overlays used in this article and reader. The source photographs and annotations retain their CC BY 4.0 terms; the method-generated numerical fields in the mixed NPZ files are separate research outputs. The first-party software license does not relicense those dataset components.

`docs/MANTA_MATERIALS.csv` and `docs/MANTA_CREDITS.md` identify the included file objects, source versions and display changes. The 31 entries are file objects, not 31 experimental samples. Complete provider datasets and weights are external. Keep provider attribution and the documented transformations attached to included scientific examples.

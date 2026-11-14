# Data, checkpoints and upstream source

Saved-result replay uses the supplied numerical observations. Original-image execution additionally needs the exact dataset members, model checkpoint and numerical source versions below. Complete dataset archives, checkpoints, feature caches and reference banks are external.

| Resource | Study identity | Provider / binding |
|---|---|---|
| DINOv2 ViT-S/14 without registers | SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | [Official checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth); `weights.anomalydino`. |
| VisA archive | 1,929,840,640 bytes; SHA256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` | [Original tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar); `image_sources.VisA`. |
| Native MANTA paddy archive | 2,759,300,051 bytes; SHA256 `32df00cc1c8fb41a140d6607f49a1dee9de0fabf7c977bdda89f235584cc8c14` | [Provider file](https://ndownloader.figshare.com/files/50906889); `image_sources.MANTA`. Tiny resizes do not replace these five-view composites. |
| DINOv2 source | Commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | [Official repository](https://github.com/facebookresearch/dinov2). |
| AnomalyDINO source | Commit `b9d1c2648e3a5247437d4d953d907a8f3d994457` | [Official repository](https://github.com/dammsi/AnomalyDINO). |
| PatchCore source | Commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`; sampler SHA256 `39612cd2b486865ece304348f740f4bc700a1c2e8684fbab3c2ed016c7801f2d` | [Official repository](https://github.com/amazon-science/patchcore-inspection). |
| Wide ResNet-50-2 V2 | SHA256 `9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950` | [TorchVision checkpoint](https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth); `mhpc_torch_hub`. |
| Giant DINOv2 with registers | Revision `8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb` | [Model repository](https://huggingface.co/facebook/dinov2-with-registers-giant/tree/8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb); file identities in `protocol/model_pins.json`. |

The last two checkpoints serve their respective complete competing systems, not the selected ViT-S directional/native route. Upstream repositories are external resources; `third_party/` contains their applicable licenses and notices. The supplied method adapters are distinct from a complete official checkout. Exact-checkout validation requires the full specified commit, including members listed in its source inventory. A Git LFS pointer is not a checkpoint or a bank.

## Bind resources

Use a copy of `RESOURCE_BINDINGS.example.json` outside the package. `weights.anomalydino` names the checkpoint file; `image_sources` names an original archive or an extracted directory preserving original member names and bytes. `mhpc_torch_hub` and `subspace_giant` are optional unless selecting their corresponding competitor. Relative paths resolve against the binding file's directory.

`RESOURCE_PATHS.example.json` is the source/model-loader binding: its `upstream` entries point to exact official checkouts. It is not the image-membership binding above. A file path resolves a local resource; a source commit or SHA256 identifies the resource expected by the study. Keep these roles distinct.

Dataset memberships identify fit-normal, calibration-normal and query roles, their order, views, crops and source bytes. Do not exchange Tiny/native MANTA images or infer member identity from a filename alone. Query masks are opened for evaluation after the required predictions are complete.

Soybean/terminal use Tiny agriculture file 50971437 (1,066,801,017 bytes, SHA256 `6935e44bcf8dca04fff9dd045f307d802356a097ccf71a9096d450136fe3743d`) and mechanics file 50971446 (3,197,059,395 bytes, SHA256 `57d5a326cd24119cfa5406a7c267931c6840f9bd1f3c71c1599cf119290048bc`). Their separate resource bindings are described in `extensions/unseen/README.md`.

## Included scientific examples and terms

`figure_sources/shared/data/` contains selected photographs, GT, crops and scientific fields/arrays used to inspect the reported examples, including negative cases. These are research inputs, not the complete datasets. `docs/MANTA_MATERIALS.csv` identifies the selected MANTA file objects; 31 entries are 26 photographs/derivatives and 5 NPZ objects, not 31 experimental samples.

VisA's provider dataset license is CC BY 4.0. The selected MANTA photographs and source annotations use CC BY 4.0 from the four exact Figshare versions listed in `MANTA_CREDITS.md`. The license permits reproduction, adaptation and redistribution with attribution, a license link and identification of changes. The article and reader preserve the dataset authors, version DOIs, source object identities and display modifications. The source-license column in `MANTA_MATERIALS.csv` applies to source photographs and annotations; method-generated fields in mixed NPZ objects are separate research outputs. The first-party software license applies to software contributions only; dataset, checkpoint and third-party terms remain separate.

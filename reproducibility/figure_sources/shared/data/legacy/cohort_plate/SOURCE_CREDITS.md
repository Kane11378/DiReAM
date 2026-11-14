# Source photographs and credits

The component displays 19 complete native views from seven fixed normal
reference inspection units. All files are identified individually in
`NATIVE_VIEWS.csv`; `SOURCE_CREDITS.json` records their exact original dataset
archive members and source/pixel identities. The selection is the first
retained normal reference in each category's fixed cohort order, not a choice
based on query outcomes.

**MANTA.** Lei Fan, Dongdong Fan, Zhiguang Hu, Yiwen Ding, Donglin Di, Kai Yi,
Maurice Pagnucco and Yang Song. *MANTA: A Large-Scale Multi-View and
Visual-Text Anomaly Detection Dataset for Tiny Objects*. CVPR, 2025,
pp. 25518–25527. [Dataset project](https://grainnet.github.io/MANTA/),
[source paper](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_MANTA_A_Large-Scale_Multi-View_and_Visual-Text_Anomaly_Detection_Dataset_for_CVPR_2025_paper.html).
Paddy, maize and square button cap each contribute five acquired views of
one provider physical object. The selected source photographs and annotations use
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) attached to the
exact Figshare versions listed below. Reproduction, adaptations and
redistribution retain the dataset authors, version source, license link
and description of display changes.

**VisA.** Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar
Dabeer. *SPot-the-Difference Self-supervised Pre-training for Anomaly Detection
and Segmentation*. ECCV, 2022, pp. 392–408.
[Source paper](https://doi.org/10.1007/978-3-031-20056-4_23),
[official source](https://github.com/amazon-science/spot-diff).
Cashew, capsules, pipe fryum and PCB1 each contribute one dataset image; an
image may contain multiple physical instances. The official source states
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The PNG assets preserve the decoded source RGB values. For MANTA, the
listed complete-view boxes split the acquired slots from a native composite;
for VisA the asset is the complete original image decoded into RGB. No extra
crop, rotation, contrast, gamma, brightness or colour change is applied.
Whole native views are displayed proportionally with white letterboxing.
The labels, counts and layout are authored for this study; the source
photographs retain the dataset credits and terms above.

## Version-specific selected MANTA objects

Selected MANTA photographs and source annotations are credited to Lei Fan, Dongdong Fan, Zhiguang Hu, Yiwen Ding, Donglin Di, Kai Yi, Maurice Pagnucco and Yang Song, authors of MANTA (CVPR 2025). The source releases used here are Tiny 27934938.v3, native paddy 27938454.v1, native maize 27935718.v1 and native square button cap 27937521.v1. Their version-specific records identify Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/ . The four version DOIs are https://doi.org/10.6084/m9.figshare.27934938.v3 ; https://doi.org/10.6084/m9.figshare.27938454.v1 ; https://doi.org/10.6084/m9.figshare.27935718.v1 ; https://doi.org/10.6084/m9.figshare.27937521.v1 . The selected MANTA photographs and source annotations identified in `docs/MANTA_MATERIALS.csv` are reproduced and distributed under the CC BY 4.0 license attached to these exact Figshare versions. That license permits reproduction, adaptation and redistribution with attribution, a license link and identification of changes, including the crops, resizing and overlays used in this article and reader. The source photographs and annotations retain their CC BY 4.0 terms; the method-generated numerical fields in the mixed NPZ files are separate research outputs. The first-party software license does not relicense those dataset components.

Figure S1 uses one complete Tiny paddy view with explanatory footprint overlays. Figure S2 separates five acquired native views per MANTA object and uses proportional display with white letterboxing. The three MANTA cases in Figure S12 show complete selected native views, source GT and derived method-wise rankings computed over all acquired views; the 102-square enlarged ROI in that figure belongs to VisA cashew058. In Figure S15 the first RGB column preserves the native aspect ratio; the overlay columns use RGB resized bicubically to 256 square, joint ranks over all five views, an upper-5% viridis overlay at alpha 0.82, and black/white GT outlines. These are display operations on retained observations, not new predictions or altered evaluation GT.

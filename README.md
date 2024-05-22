### Pyscanpath

Università degli Studi di Milano

Course: Natural Interaction

Student: Matteo Castagna

Registration number: 27366A

Email: matteo.castagna2@studenti.unimi.it

Pyscanpath is a framework to easily load and test few scanpath prediction models and to compare their result to a ground truth through some common metrics for scanpath evaluation.

---

**Models**

Four models for scanpath prediction are available:
- __[Itti-Koch](https://github.com/DirkBWalther/SaliencyToolbox)__ (original in MATLAB, here partially rewritten in python)
- __[Constrained Levy Exploration (CLE)](https://github.com/phuselab/CLE)__
- __[DeepGazeIII](https://github.com/matthias-k/DeepGaze)__
- __[IOR-ROI Recurrent Mixture Density Network](https://github.com/sunwj/scanpath)__

---

**Metrics**

Four metrics to compute distances or similarity between scanpaths are available:
- Euclidean distance
- Mannan distance
- Edit distance
- Time delay embedding

Edit distance and time delay embedding implementations are taken from __[FixaTons](https://github.com/dariozanca/FixaTons/tree/master)__.

---

**Datasets**
Pyscanpath let test the models on the MIT1003 dataset, the CAT3000 dataset and the OSIE dataset which can be accessed via the datasets module. This functionality is implemented upon __[pysaliency](https://github.com/matthias-k/pysaliency/tree/dev)__ which needs to be installed.
The models can be tested on images and scanpaths, provided as numpy arrays, singularly too, as can be seen in demo.ipynb.

---

### Download Pyscanpath

- Clone or download the folder with the code
- Download data.zip from [here](https://mega.nz/file/KvxEXS5Z#p-ZxpjiJ6k9Tj9vxH8CGX0Ec9MQW0SJX_XSeEJcmvW0), extract the data folder and place it in `pyscanpath\models\Iorroi`
- IOR-ROI model to work requires Meta [SAM](https://github.com/facebookresearch/segment-anything). After installing SAM, download vit_h.pth from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), create a folder named checkpoint in `pyscanpath\models\Iorroi` and place there the .pth file.
- DeepgazeIII in order to work requires [pysaliency](https://github.com/matthias-k/pysaliency/tree/dev)


![](/sample/example.png)

*Examples of predicted scanpath and ground truth (stimuli from MIT1003)*

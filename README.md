# DLG: Feature Inference Attacks for CIFAR-100. 

## How to run?

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ./dlg.py --help
```

## Copyright Notice

This work is an implementation of the two papers:

1. Deep Leakage from Gradients:
```txt
@inproceedings{zhu19deep,
  title={Deep Leakage from Gradients},
  author={Zhu, Ligeng and Liu, Zhijian and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

2. Improved Deep Leakage from Gradients: 
```txt
@article{zhao2020idlg,
  title={idlg: Improved deep leakage from gradients},
  author={Zhao, Bo and Mopuri, Konda Reddy and Bilen, Hakan},
  journal={arXiv preprint arXiv:2001.02610},
  year={2020}
}
```

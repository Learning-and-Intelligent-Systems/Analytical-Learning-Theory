# Towards Generalization in Machine Learning via Analytical Learning Theory

This repo contains the code for reproducing the results in our paper:

**Towards Generalization in Machine Learning via Analytical Learning Theory** \
*Kenji Kawaguchi, Yoshua Bengio, Vikas Verma, and Leslie Pack Kaelbling* \
https://arxiv.org/pdf/1802.07426.pdf

```bibtex
@article{kawaguchi2018generalization,
  title={Generalization in Machine Learning via Analytical Learning Theory},
  author={Kawaguchi, Kenji and Bengio, Yoshua and Verma, Vikas and Kaelbling, Leslie Pack},
  journal={arXiv preprint arXiv:1802.07426},
  year={2018}
}
```

### Requirements
This code has been tested with  
python 2.7.9  
torch 0.3.1  
torchvision 0.2.0


Add the --temp_dir and --home_dir as appropriate in the following commands.
For Cifar10 and Cifar100, we used --cutsize 16 and for SVHN we used --cutsize 20


### How to run DualCutout
```
python cifar10/resnext/main.py --dualcutout --dataset cifar10 --arch wrn28_10 \
--epochs 300 --batch_size 64 --learning_rate 0.1 --data_aug 1 --decay 0.0005 --schedule 150 225 \
--gamma 0.1 0.1 --alpha 0.1 --cutsize 16
```

### How to run Single Cutout
```
python cifar10/resnext/main.py --singlecutout --dataset cifar10 --arch wrn28_10 \
--epochs 300 --batch_size 64 --learning_rate 0.1 --data_aug 1 --decay 0.0005 --schedule 150 225 \
--gamma 0.1 0.1 --alpha 0.1 --cutsize 16
```
### How to run baseline
```
python cifar10/resnext/main.py --dataset cifar10 --arch wrn28_10 \
--epochs 300 --batch_size 64 --learning_rate 0.1 --data_aug 1 --decay 0.0005 --schedule 150 225 \
--gamma 0.1 0.1
```

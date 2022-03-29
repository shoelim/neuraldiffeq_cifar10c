### Comparing robustness of deep ConvMixer and Neural ODE trained on CIFAR10

Their robustness is evaluated on CIFAR10-C

With deep ConvMixer:
'''
python odenet_cifar10.py --dim 64 --network resnet --downsampling-method conv 
'''

With Neural ODE:
'''
python odenet_cifar10.py --dim 64 --network odenet --downsampling-method conv
''' 

# cpp_torch
## [libtorch](https://pytorch.org/get-started/locally/) at Visual Studio C++  
####header only, dependency-free deep learning framework  

Directory structure
<img src="./images/image00.png"/>  

This project aims to be a wrapper for libtorch to make [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) compatible with GPU.
[**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) really great, but unfortunately it can not be calculated on a GPU.
At first glance, this header-only framework aims to be used as written in [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn).


## Requirements  
visual studio 2015,2017  

[libtorch](https://pytorch.org/get-started1locally/)
**Please adapt the version of cuda to your environment**  
<img src="./images/image01.png"/>  

## tiny-dnn
[BSD 3-Clause License Copyright (c) 2013, Taiga Nomi](https://github.com/tiny-dnn/tiny-dnn)  
Used for nonlinear regression and nonlinear time series prediction
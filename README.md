# [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) based on  [libtorch](https://pytorch.org/get-started/locally/) 

**header only, deep learning framework with no dependencies other than libtorch**  

Directory structure
<img src="./images/image00.png"/>  

This project aims to be a wrapper for libtorch to make [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) compatible with GPU.
[**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) really great, but unfortunately it can not be calculated on a GPU.
At first glance, this header-only framework aims to be used as written in [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn).

<img src="./images/image02.png"/>  

**progress**  
*tiny_dnn*  
```
tiny_dnn::progress_display disp(train_images.size());
```
  
*cpp_torch*
```
cpp_torch::progress_display disp(train_images.size());
```

cpp_torch::progress_display disp(train_images.size());  
<img src="./images/colored_progress.png"/>  

**data set download**  
<img src="./images/url_file_download.png" width=60%>


## Requirements  
visual studio 2015,2017  

[libtorch](https://pytorch.org/get-started1locally/)
**Please adapt the version of cuda to your environment**  


## tiny-dnn
[BSD 3-Clause License Copyright (c) 2013, Taiga Nomi](https://github.com/tiny-dnn/tiny-dnn)  

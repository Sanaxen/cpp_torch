
### Enabled to build with **libtorch1.5**  
### CUDA 9.2 is required for calculation using GPU.  

!~~!Currently, the cause is unknown, but the result is incorrect in CPU mode of libtorch 1.5.~~  
It's not a smart method, but it has been fixed to give the correct result.  

**Infomation**  
[libtorch]Same model in CUDA and CPU got different result? [#26456](https://github.com/pytorch/pytorch/issues/26456)  


### **can solve the problem that is not working on GPU (CUDA) (torch::cuda::is_available() returns false).**  

####How to resolve this issue  
add the following argument -INCLUDE:?warp_size@cuda@at@@YAHXZ  
**Linker -> Command line -> Addtitional options  
-INCLUDE:?warp_size@cuda@at@@YAHXZ**  

#### reference  
Windows Libtorch Ver1.5 did't operate at the GPU mode , it operated at the CPU mode. [#37568](https://github.com/pytorch/pytorch/issues/37568)  
Libtorch 1.5 failed to access CUDA devices [#37481](https://github.com/pytorch/pytorch/issues/37481)  

~~The current code works with **libtorch1.3**, but the latest is **libtorch1.4**.~~

~~Support for libtorch1.3 to libtorch1.4 is being promoted. Please wait.~~
The current changes that we learned from the work It cannot be compiled with **visual studio2015**.  
**visual studio2017** also requires updates to the latest patches.

``with_bias-> bias``
transposed seems to be abolished in 
``torch :: nn :: Conv2dOptions. ``
Instead Must be changed to 
``torch :: nn :: ConvTranspose2d.``  

``torch :: nn :: FeatureDropout``
also seems to be abolished. instead of Use 
``torch :: nn :: Dropout2d.``  

Other  
``torch :: Tensor & tensor = torch :: tensor ({vec [i]});``  
Is said to be a tensor and terminates abnormally.  
``torch :: Tensor & tensor = torch :: tensor (vec [i]);``  
Must be.  

 (/std:c++17)  
std::byte compile error C2872  
compile option -> ``/D _HAS_STD_BYTE=0``  

Linker -> Command line -> Addtitional options  
-INCLUDE:?warp_size@cuda@at@@YAHXZ  
* * *

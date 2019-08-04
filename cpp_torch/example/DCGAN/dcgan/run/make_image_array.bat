:copy C:\dev\opencv-3.4.0\build\x64\vc15\bin\*.dll /v /y

image_concat.exe generated_images 8x8 test_%1.bmp
del /Q generated_images\*.bmp

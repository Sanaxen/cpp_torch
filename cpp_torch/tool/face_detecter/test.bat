set MODEL=.\face_detector\res10_300x300_ssd_iter_140000.caffemodel
set PROTO=.\face_detector\deploy.prototxt

:set image=input_image\video-of-people-walking-855564.mp4
:set image=input_image\sample.png
set image=input_image

:image file
.\x64\Release\resnet_ssd_face.exe --min_confidence=0.6 --model=%MODEL% --proto=%PROTO% --video=%image%

: camer
:.\x64\Release\resnet_ssd_face.exe --min_confidence=0.6 --model=%MODEL% --proto=%PROTO%

:pause
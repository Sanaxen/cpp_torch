set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
#set yrang[-1:20]
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
#set size ratio -1

#unset key

file1 = "training.dat"

plot file1 using 1:3   t "Observation"  with lines linewidth 2 lc "web-blue"
replot file1 using 1:5   t "Observation"  with lines linewidth 2 lc "web-blue"
replot file1 using 1:7   t "Observation"  with lines linewidth 2 lc "web-blue"

replot file1 using 1:2   t "predict"  with lines linewidth 2 lc "orange-red"
replot file1 using 1:4   t "predict"  with lines linewidth 2 lc "orange-red"
replot file1 using 1:6   t "predict"  with lines linewidth 2 lc "orange-red"

file2 = "predict.dat"

replot file2 using 1:3   t "Observation"  with lines dt (10,5) linewidth 2 lc "web-blue"
replot file2 using 1:5   t "Observation"  with lines dt (10,5) linewidth 2 lc "web-blue"
replot file2 using 1:7   t "Observation"  with lines dt (10,5) linewidth 2 lc "web-blue"

replot file2 using 1:2   t "predict"  with lines linewidth 2 lc "orange-red"
replot file2 using 1:4   t "predict"  with lines linewidth 2 lc "orange-red"
replot file2 using 1:6   t "predict"  with lines linewidth 2 lc "orange-red"

#set terminal png
#set out "image1.png"
#replot

#set terminal windows
#set output

pause 3
reread

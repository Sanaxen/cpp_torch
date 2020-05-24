set encoding utf8
set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set term windows size 640,480
set term pngcairo size 640,480
set output "fitting.png"
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
#set size ratio -1

unset key
#set xdata time
#set timefmt "%Y/%m/%d[%H:%M:%S]"
#set xtics timedate
#set xtics format "%Y/%m/%d"

file = "test.dat"

plot file using 1:2   t "observation"  with lines linewidth 2 lc "green",\
"predict1.dat" using 1:2   t "observation"  with lines linewidth 2 lc "blue",\
"predict2.dat" using 1:2   t "observation"  with lines linewidth 2 lc "web-blue",\
"prophecy.dat" using 1:2   t "prophecy"  with lines linewidth 2 lc "web-blue" dt 3

replot file using 1:3   t "predict"  with lines linewidth 2 lc "green",\
"predict1.dat" using 1:3   t "predict"  with lines linewidth 2 lc "red",\
"predict2.dat" using 1:3   t "predict"  with lines linewidth 2 lc "plum",\
"prophecy.dat" using 1:3   t "prophecy"  with lines linewidth 2 lc "magenta"


replot file using 1:4   t "observation"  with lines linewidth 2 lc "green",\
"predict1.dat" using 1:4   t "observation"  with lines linewidth 2 lc "blue",\
"predict2.dat" using 1:4   t "observation"  with lines linewidth 2 lc "web-blue",\
"prophecy.dat" using 1:4   t "prophecy"  with lines linewidth 2 lc "web-blue" dt 3

replot file using 1:5   t "predict"  with lines linewidth 2 lc "green",\
"predict1.dat" using 1:5   t "predict"  with lines linewidth 2 lc "red",\
"predict2.dat" using 1:5   t "predict"  with lines linewidth 2 lc "plum",\
"prophecy.dat" using 1:5   t "prophecy"  with lines linewidth 2 lc "magenta"



replot file using 1:6   t "observation"  with lines linewidth 2 lc "green",\
"predict1.dat" using 1:6   t "observation"  with lines linewidth 2 lc "blue",\
"predict2.dat" using 1:6   t "observation"  with lines linewidth 2 lc "web-blue",\
"prophecy.dat" using 1:6   t "prophecy"  with lines linewidth 2 lc "web-blue" dt 3

replot file using 1:7   t "predict"  with lines linewidth 2 lc "green",\
"predict1.dat" using 1:7   t "predict"  with lines linewidth 2 lc "red",\
"predict2.dat" using 1:7   t "predict"  with lines linewidth 2 lc "plum",\
"prophecy.dat" using 1:7   t "prophecy"  with lines linewidth 2 lc "magenta"


set term windows size 640,480
set term pngcairo size 640,480
set output "fitting.png"
replot


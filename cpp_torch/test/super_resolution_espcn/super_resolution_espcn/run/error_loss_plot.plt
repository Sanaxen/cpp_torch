set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid

set yrange[230:300]

plot 'loss.dat' using 1   t "loss"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1
replot 'loss.dat' using 2   t "loss"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1
replot 'loss.dat' using 1  smooth bezier t "G"  with lines linewidth 2 linecolor rgbcolor "red"
replot 'loss.dat' using 2  smooth bezier t "D"  with lines linewidth 1 linecolor rgbcolor "blue"

pause 5
reread

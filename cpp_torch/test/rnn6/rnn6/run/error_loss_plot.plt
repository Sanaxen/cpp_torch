set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid


plot 'error_loss.dat' using 1   t "loss"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1
replot 'error_loss.dat' using 1  smooth bezier t "loss"  with lines linewidth 2 linecolor rgbcolor "red"
replot 'error_loss.dat' using 2  smooth bezier t "best"  with lines linewidth 1 linecolor rgbcolor "blue"
replot 'error_loss.dat' using 3  smooth bezier t "goal"  with lines linewidth 1 linecolor rgbcolor "green"

replot 'error_var_loss.dat' using 1  smooth bezier t "vari"  with lines linewidth 2 linecolor rgbcolor "magenta" dt 13

pause 5
reread

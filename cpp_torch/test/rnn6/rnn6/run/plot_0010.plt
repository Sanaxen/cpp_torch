set encoding utf8
set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set colorsequence podo
set term windows size 1920,1440
set term pngcairo size 1920,1440
set output "observed_predict_NL.png"
set style circle radius graph 0.010000
set style fill  transparent solid 0.35 noborder
set datafile separator ","
set nokey
set palette rgbformulae 22, 13, -31
set xlabel "observed"
set ylabel "predict"
#set style circle radius graph 0.005
plot 'plot_0010(000).dat'  using 1:2:3 t "predict" with circles lw 0.1 pal
set label 1 at graph 0.50,0.85 "observed=predict"
set datafile separator ","
replot 'plot_0010(001).dat'  using 1:2 t "observed" with lines linewidth 2.0 
set term windows size 1920,1440
set term pngcairo size 1920,1440
set output "observed_predict_NL.png"
replot
unset multiplot
pause 0

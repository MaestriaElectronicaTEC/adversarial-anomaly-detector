#!/bin/gnuplot

reset

# Definición de intervalo y de función

si(x)=(x==0)?1:sin(x)/x
tau=1

DE=-3.2
A=3.2
MINY=-0.3*tau
MAXY=1.05*tau
f(x)=tau*si(pi*x*tau)

# Número de muestras

N=A-DE+1
set samples N

# Dónde escribir y cómo

set terminal postscript eps color 18 
set output "prototipo_gnuplot_.eps"

# Ejes y su etiquetación

unset xlabel
unset ylabel
set xrange[DE:A]
set yrange[MINY:MAXY]
set xtics axis 1
set ytics axis 1
set format x "%g"
set format y " "

set label "F" at A+0.2,0
set label "X(F)" at -0.8,MAXY norotate

unset border
set xzeroaxis linetype -1 linewidth 1
set yzeroaxis linetype -1 linewidth 1

set multiplot
set samples N*10

plot f(x) notitle with lines lt -1

unset multiplot
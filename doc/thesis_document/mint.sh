#!/bin/bash

if test `whoami` != "root" 
then
  echo "  Error: You need root priviledges to run this script."
  exit $FAILURE
fi

if [ -f /usr/bin/jpeg2ps ]
then
    echo "jpeg2ps is installed."
else
    echo "Moving jpeg2ps to the system's path"
    cp jpeg2ps /usr/bin/jpeg2ps
fi

apt-get install \
    texlive \
    texlive-base \
    texlive-binaries \
    texlive-common \
    texlive-extra-utils \
    texlive-fonts-recommended \
    texlive-font-utils \
    texlive-formats-extra \
    texlive-generic-recommended \
    texlive-lang-spanish \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-math-extra \
    texlive-plain-extra \
    texlive-pstricks \
    texlive-science \
    texlive-bibtex-extra \
    texlive-publishers \
    xfig \
    transfig \
    imagemagick \
    ps2eps \
    pdftk \
    bibutils \
    biblatex \
    gnuplot \
    octave

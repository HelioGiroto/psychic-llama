mkdir ./biology
cd biology
echo http://biology.as.nyu.edu/object/${1}.html | wget -O- -i- | hxnormalize -x  | hxselect -i p | lynx -stdin -dump -nonumbers -hiddenlinks=ignore > ${1}




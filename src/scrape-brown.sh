mkdir ./${1}
cd $1
echo $2 | wget -O- -i- | hxnormalize -x  | hxselect -i li | lynx -stdin -dump -nonumbers -hiddenlinks=ignore >${1}-labs

cat ${1}-labs | grep -v "file" | grep -v "http" | grep -v "mail" | egrep -v '^[[:space:]]*$' | grep -v "Research"  | grep -v "Lab" | awk 'NF<=3' | grep -v "@" | grep -v "Professor" > ${1}-labs-cleaned


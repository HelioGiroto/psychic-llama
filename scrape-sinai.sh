mkdir ./${1}
cd $1
echo "http://icahn.mssm.edu/research/labs" | wget -O- -i- | hxnormalize -x  | hxselect -i ul | lynx -stdin  -dump -nonumbers -hiddenlinks=ignore | grep -i "http" > ${1}-labs

cat ${1}-labs | while read line
do
	IFS='/' read -a array<<< $line
	filename="${array[0]##* }"
	echo $line | wget -O- -i- | hxnormalize -x | hxselect -i p | lynx -stdin -dump -nonumbers > $filename

done

cat $(ls) > combined 
cat $(ls) | grep  -v "http" | grep -v "file" | grep -v ";" | iconv -f utf-8 -t ascii//translit > cleaned-combined

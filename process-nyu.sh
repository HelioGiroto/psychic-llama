cat $(ls) > combined
cat $(ls) | grep  -v "http" | grep -v "file" | grep -v ";" | iconv -f utf-8 -t ascii//translit > cleaned-combined

#!/usr/bin/env bash

for f in $(find . -type f -iname '*.svg'); do
    echo "Converting $f"
    inkscape --file="$f" --export-pdf="${f%%.svg}".pdf
done

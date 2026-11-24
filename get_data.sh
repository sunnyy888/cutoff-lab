#!/usr/bin/env bash
# Download the public hotel-booking dataset (Antonio, Almeida & Nunes 2019) from the TidyTuesday mirror.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
curl -sSL -o data/hotels.csv https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv
echo "saved data/hotels.csv ($(wc -l < data/hotels.csv) lines)"

#!/usr/bin/env bash
# Fetch the ICRAF-ISRIC Soil VNIR Spectral Library from the ICRAF Dataverse.
# Public, no registration. Note: the server does not support byte ranges, so an
# interrupted download must be restarted rather than resumed.
set -euo pipefail
BASE=https://data.worldagroforestry.org/api/access/datafile
mkdir -p data && cd data
curl -L -o ASD_Spectra.tab         "$BASE/11274"   # Vis-NIR reflectance, 9.1 MB
curl -L -o Chemical_properties.tab "$BASE/11275"   # ORGC and other chemistry
curl -L -o Country.tab             "$BASE/11283"   # ISO -> country -> REGION
curl -L -o Region.tab              "$BASE/11297"   # REGION code -> name
curl -L -o Site_description.tab    "$BASE/11300"   # coordinates, altitude
curl -L -o ICRAF_sample_codes.tab  "$BASE/11290"   # Batch_Labid <-> Sampleno
echo "downloaded to $(pwd)"

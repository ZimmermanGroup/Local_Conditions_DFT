#!/bin/bash

# Fetch the content of the webpage
url="http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/CHARGE_MULTIPLICITY_G2RC.txt"
web_content=$(curl -s "$url")

# Filter out the molecules with zero charge and singlet (spin = 0) multiplicity
selected_molecules=$(echo "$web_content" | grep -E '^[A-Za-z0-9]+\s+0\s+1$' | awk '{print $1}')

# Python script command
python_script="Score2_arg.py"

# Charge, spin, correlation, and exchange values
charge=0
spin=0
correlation="GGA_C_SOGGA11"
exchange="GGA_X_SOGGA11"

# Loop through the selected molecules and run the Python script
for molecule_name in $selected_molecules; do
    output_file="$molecule_name.txt"
    echo "Running $python_script for $molecule_name..."
    python3 "$python_script" "$molecule_name" $charge $spin $correlation $exchange > "$output_file"
    echo "Completed $molecule_name. Output saved to $output_file"
done

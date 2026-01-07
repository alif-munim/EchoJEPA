#!/bin/bash
# Fix DOS line endings in SLURM and shell scripts

echo "Converting line endings to Unix format..."

# Convert the first100 script
if [[ -f "dicom_batch_convert_first100.slurm" ]]; then
  dos2unix dicom_batch_convert_first100.slurm 2>/dev/null || \
    sed -i 's/\r$//' dicom_batch_convert_first100.slurm
  echo "✓ Fixed dicom_batch_convert_first100.slurm"
fi

# Convert the classification script
if [[ -f "classify_first100.slurm" ]]; then
  dos2unix classify_first100.slurm 2>/dev/null || \
    sed -i 's/\r$//' classify_first100.slurm
  echo "✓ Fixed classify_first100.slurm"
fi

# Convert the analysis script
if [[ -f "analyze_timing_first100.sh" ]]; then
  dos2unix analyze_timing_first100.sh 2>/dev/null || \
    sed -i 's/\r$//' analyze_timing_first100.sh
  chmod +x analyze_timing_first100.sh
  echo "✓ Fixed analyze_timing_first100.sh"
fi

# Convert the classification readiness check
if [[ -f "check_classification_ready.sh" ]]; then
  dos2unix check_classification_ready.sh 2>/dev/null || \
    sed -i 's/\r$//' check_classification_ready.sh
  chmod +x check_classification_ready.sh
  echo "✓ Fixed check_classification_ready.sh"
fi

# Convert cohort test script
if [[ -f "test_cohort_single_study.sh" ]]; then
  dos2unix test_cohort_single_study.sh 2>/dev/null || \
    sed -i 's/\r$//' test_cohort_single_study.sh
  chmod +x test_cohort_single_study.sh
  echo "✓ Fixed test_cohort_single_study.sh"
fi

# Convert cohort conversion script
if [[ -f "convert_cohorts123.sh" ]]; then
  dos2unix convert_cohorts123.sh 2>/dev/null || \
    sed -i 's/\r$//' convert_cohorts123.sh
  chmod +x convert_cohorts123.sh
  echo "✓ Fixed convert_cohorts123.sh"
fi

# Convert cohort classification script
if [[ -f "classify_cohorts123.py" ]]; then
  dos2unix classify_cohorts123.py 2>/dev/null || \
    sed -i 's/\r$//' classify_cohorts123.py
  chmod +x classify_cohorts123.py
  echo "✓ Fixed classify_cohorts123.py"
fi

# Convert cohorts 1-9 classification script
if [[ -f "classify_cohorts1-9.py" ]]; then
  dos2unix classify_cohorts1-9.py 2>/dev/null || \
    sed -i 's/\r$//' classify_cohorts1-9.py
  chmod +x classify_cohorts1-9.py
  echo "✓ Fixed classify_cohorts1-9.py"
fi

# Convert deidentification script
if [[ -f "deidentify_results.py" ]]; then
  dos2unix deidentify_results.py 2>/dev/null || \
    sed -i 's/\r$//' deidentify_results.py
  chmod +x deidentify_results.py
  echo "✓ Fixed deidentify_results.py"
fi

# Convert cohort classification SLURM script
if [[ -f "classify_cohorts123.slurm" ]]; then
  dos2unix classify_cohorts123.slurm 2>/dev/null || \
    sed -i 's/\r$//' classify_cohorts123.slurm
  echo "✓ Fixed classify_cohorts123.slurm"
fi

# Convert cohort conversion SLURM script
if [[ -f "convert_cohorts123.slurm" ]]; then
  dos2unix convert_cohorts123.slurm 2>/dev/null || \
    sed -i 's/\r$//' convert_cohorts123.slurm
  echo "✓ Fixed convert_cohorts123.slurm"
fi

# Convert cohort1 array SLURM script
if [[ -f "convert_cohort1_array.slurm" ]]; then
  dos2unix convert_cohort1_array.slurm 2>/dev/null || \
    sed -i 's/\r$//' convert_cohort1_array.slurm
  echo "✓ Fixed convert_cohort1_array.slurm"
fi

# Convert cohort2 array SLURM script
if [[ -f "convert_cohort2_array.slurm" ]]; then
  dos2unix convert_cohort2_array.slurm 2>/dev/null || \
    sed -i 's/\r$//' convert_cohort2_array.slurm
  echo "✓ Fixed convert_cohort2_array.slurm"
fi

# Convert cohort3 array SLURM script
if [[ -f "convert_cohort3_array.slurm" ]]; then
  dos2unix convert_cohort3_array.slurm 2>/dev/null || \
    sed -i 's/\r$//' convert_cohort3_array.slurm
  echo "✓ Fixed convert_cohort3_array.slurm"
fi

# Convert cohorts 1-9 array SLURM script
if [[ -f "convert_cohorts1-9_array.slurm" ]]; then
  dos2unix convert_cohorts1-9_array.slurm 2>/dev/null || \
    sed -i 's/\r$//' convert_cohorts1-9_array.slurm
  echo "✓ Fixed convert_cohorts1-9_array.slurm"
fi

# Convert classification array SLURM script
if [[ -f "classify_cohorts123_array.slurm" ]]; then
  dos2unix classify_cohorts123_array.slurm 2>/dev/null || \
    sed -i 's/\r$//' classify_cohorts123_array.slurm
  echo "✓ Fixed classify_cohorts123_array.slurm"
fi

# Convert cohorts 1-9 classification array SLURM script
if [[ -f "classify_cohorts1-9_array.slurm" ]]; then
  dos2unix classify_cohorts1-9_array.slurm 2>/dev/null || \
    sed -i 's/\r$//' classify_cohorts1-9_array.slurm
  echo "✓ Fixed classify_cohorts1-9_array.slurm"
fi

# Convert merge classification results script
if [[ -f "merge_classification_results.sh" ]]; then
  dos2unix merge_classification_results.sh 2>/dev/null || \
    sed -i 's/\r$//' merge_classification_results.sh
  chmod +x merge_classification_results.sh
  echo "✓ Fixed merge_classification_results.sh"
fi

# Convert CPU test script
if [[ -f "test_cpu_mode.py" ]]; then
  dos2unix test_cpu_mode.py 2>/dev/null || \
    sed -i 's/\r$//' test_cpu_mode.py
  chmod +x test_cpu_mode.py
  echo "✓ Fixed test_cpu_mode.py"
fi

# Convert CPU test SLURM script
if [[ -f "test_cpu_pytorch.slurm" ]]; then
  dos2unix test_cpu_pytorch.slurm 2>/dev/null || \
    sed -i 's/\r$//' test_cpu_pytorch.slurm
  echo "✓ Fixed test_cpu_pytorch.slurm"
fi

# Convert CPU PyTorch setup script
if [[ -f "setup_cpu_pytorch.sh" ]]; then
  dos2unix setup_cpu_pytorch.sh 2>/dev/null || \
    sed -i 's/\r$//' setup_cpu_pytorch.sh
  chmod +x setup_cpu_pytorch.sh
  echo "✓ Fixed setup_cpu_pytorch.sh"
fi

# Convert main SLURM script if needed
if [[ -f "dicom_batch_convert.slurm" ]]; then
  dos2unix dicom_batch_convert.slurm 2>/dev/null || \
    sed -i 's/\r$//' dicom_batch_convert.slurm
  echo "✓ Fixed dicom_batch_convert.slurm"
fi

# Convert main shell script if needed
if [[ -f "dicom_batch_convert.sh" ]]; then
  dos2unix dicom_batch_convert.sh 2>/dev/null || \
    sed -i 's/\r$//' dicom_batch_convert.sh
  chmod +x dicom_batch_convert.sh
  echo "✓ Fixed dicom_batch_convert.sh"
fi

echo ""
echo "Done! You can now run:"
echo "  bash test_cohort_single_study.sh        # Test on one study"
echo "  bash convert_cohorts123.sh               # Convert all cohorts 1-3"
echo "  sbatch classify_cohorts123.slurm         # Classify cohorts 1-3"

rm -vf first_files_per_halo_ls
for file in $(find "/home/universe/c1838736/Tracers/V11-1/" -type f -name "Data_selectSnap235_targetT4.0-5.0-6.0_T4.0_25.0R75.0_flat-wrt-time.h5" -print0| xargs -0 -L1)
	do
		echo $file
		ls -hl $file >> first_files_per_halo_ls
	done
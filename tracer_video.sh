ffmpeg -r 1 -start_number 221 -i Data_selectSnap235_targetT4.0-5.0-6.0_T4.0_Tracer_Subset_Plot_%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 halo_6_T4.0_Tracer-Video.mp4

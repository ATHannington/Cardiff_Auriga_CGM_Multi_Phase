ffmpeg -r 1 -start_number 112 -i Data_selectSnap119_min112_max130_125R175_targetT4.0-5.0-6.0-6.75_T5.0_Tracer_Subset_Plot_%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 ISOTOPES_Tracer-Video.mp4

ffmpeg -r 1 -start_number 112 -i Data_selectSnap119_min112_max130_25R75_targetT4-5-6_T5_Quad_Plot_%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 ISOTOPES_T5_Quad-Plot.mp4

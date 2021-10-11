ffmpeg -r 1 -start_number 221 -i Data_selectSnap235_targetT4.0-5.0-6.0_Quad_Plot_%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 6 halo_6_Quad-Plot.mp4

ffmpeg -r 5 -start_number 10 -i Histogram2d_mass_%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 5 Histogram2d_mass_video.mp4 

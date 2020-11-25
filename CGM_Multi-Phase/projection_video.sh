ffmpeg -r 10 -start_number 112 -i Data_**%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Tracers_Video.mp4

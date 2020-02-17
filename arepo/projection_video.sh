ffmpeg -r 5 -start_number 10 -i Shaded_Cell_%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Shaded_Cell_video.mp4

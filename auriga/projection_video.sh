ffmpeg -r 5 -start_number 2500 -i Shaded_Cell_%04d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 5 Projections_video.mp4

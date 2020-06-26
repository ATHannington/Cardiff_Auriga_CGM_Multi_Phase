ffmpeg -r 10 -start_number 2500 -i Shaded_Cell_%04d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Projections_video_fast.mp4

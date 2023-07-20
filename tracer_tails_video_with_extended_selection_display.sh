#!/bin/bash

selectsnap=235
repeatselect=5

recreateVideosOnly=true

declare -a RadialDirectories=("25R75" "75R125" "125R175")
declare -a TemperatureDirectories=("T4.0" "T5.0" "T6.0")

for rdir in ${RadialDirectories[@]}
    do
        for tdir in ${TemperatureDirectories[@]}
            do
                dir="./""$rdir""/""$tdir"
                echo $dir

                if [[ $recreateVideosOnly == false ]]; then
                    for file in $(find "$dir" -type f -name "Data_selectSnap**_Tracer_Subset_Plot_Vid-Frame_**.png")
                        do
                            rm -vf $file
                        done

                    counter=0
                    for file in $(find "$dir" -type f -name "Data_selectSnap**_Tracer_Subset_Plot_**.png")
                        do

                            #echo $file
                            repeatcounter=0
                            selectpattern="Plot_"$selectsnap".png"
                            if [[ "$file" == *"$selectpattern"* ]]; then
                                while true; do
                                    if [[ "$repeatcounter" -le $repeatselect ]]; then
                                        printf -v counter "%03d" $((10#$counter+1))
                                        cp -vf $file ${file//_Tracer_Subset_Plot_**/_Tracer_Subset_Plot_Vid-Frame_}${counter}.png
                                        repeatcounter=$((repeatcounter+1))
                                    else
                                        break
                                    fi
                                done
                            else
                                printf -v counter "%03d" $((10#$counter+1))
                                cp -vf $file ${file//_Tracer_Subset_Plot_**/_Tracer_Subset_Plot_Vid-Frame_}${counter}.png
                            fi
                        done
                fi

                file=$(find "$dir" -type f -name "Data_selectSnap**_Tracer_Subset_Plot_Vid-Frame_001.png")

                videoframes=${file//_Tracer_Subset_Plot_Vid-Frame_**/_Tracer_Subset_Plot_Vid-Frame_}

                ffmpeg -r 2 -start_number 001 -i "$videoframes"%03d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 4 "Tracers_halo_26_"${tdir}"_"${rdir}"_Video.mp4"
                mv -f "Tracers_halo_26_"${tdir}"_"${rdir}"_Video.mp4" $dir"/Tracers_halo_26_"${tdir}"_"${rdir}"_Video.mp4"

            done
    done
                
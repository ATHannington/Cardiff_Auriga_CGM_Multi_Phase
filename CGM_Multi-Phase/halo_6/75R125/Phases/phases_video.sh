ffmpeg -r 1 -start_number 221 -i Tracers_selectSnap235_snap%03d_mass_PhaseDiagram_Individual-Temps.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Tracers_mass_Phases_Video.mp4

ffmpeg -r 1 -start_number 221 -i Tracers_selectSnap235_snap%03d_tcool_PhaseDiagram_Individual-Temps.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Tracers_tcool_Phases_Video.mp4

ffmpeg -r 1 -start_number 221 -i Tracers_selectSnap235_snap%03d_gz_PhaseDiagram_Individual-Temps.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Tracers_gz_Phases_Video.mp4

ffmpeg -r 1 -start_number 221 -i Tracers_selectSnap235_snap%03d_tcool_tff_PhaseDiagram_Individual-Temps.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -r 10 Tracers_tcool_tff_Phases_Video.mp4

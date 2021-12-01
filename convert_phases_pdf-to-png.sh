for f in Tracers*PhaseDiagram*.pdf;
  do echo $f;
  convert -density 80 $f ${f%.pdf}.png;
done

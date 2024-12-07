# drone-detect_plot_object
The intesion of this project is to convert detections in to gps location. This is a fairly simple translation considering fov of lens hieght of drone. 



todo:
add detection model done
find current location of drone 

find current heading of drone

translate detections to gps location done



Hardware test:

SpeedyBee F405 WING Flight Controller
jetson nano orion


dataflow:
telemetry data provided by flight controller passed over uart to jetson
jetson seperate spotter camera attached. 
likely want lte for person locaton updates



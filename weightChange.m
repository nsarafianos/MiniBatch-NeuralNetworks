function [ThetaInDegrees] = weightChange(vec1,vec2)
% Takes as an input two vectors and computes their angle using the dot
% product. 
CosTheta = dot(vec1,vec2)/(norm(vec1)*norm(vec2));
ThetaInDegrees = acos(CosTheta)*180/pi;
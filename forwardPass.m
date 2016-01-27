function [ Out_nl, Out, Ih_cell,Ih_nl_cell, TotalError ] = forwardPass(inputs, weights, current_class, param)

hiddenLayersNum = size(weights{2},2)+1;
Ih = inputs*weights{1};
Ih_nl = nonLinearity(param.nonLinearityFunc, Ih);
Ih_cell = cell(hiddenLayersNum);
Ih_nl_cell = cell(hiddenLayersNum);
if hiddenLayersNum > 1
    for i = 1 : hiddenLayersNum-1
        Ih_cell{i} = Ih;
        Ih_nl_cell{i} = Ih_nl;
        hin = [Ih_nl ones(size(inputs,1),param.bias)]; % Add bias to this layer
        h_out = hin*weights{2}{i};
        h_out_nl = nonLinearity(param.nonLinearityFunc, h_out);
        
        % Updates to save in the cells for the backpropagation
        Ih = h_out;
        Ih_nl = h_out_nl;
    end 
end
Ih_cell{hiddenLayersNum} = Ih;
Ih_nl_cell{hiddenLayersNum} = Ih_nl;
% These is true for the last hidden layer
Ih_nl = [Ih_nl ones(size(inputs,1),param.bias)]; % Bias of the hidden layer
Out = Ih_nl*weights{3}';
Out_nl = nonLinearity(param.nonLinearityFunc, Out);
Error = 0.5*(current_class-Out_nl).^2;

TotalError = sum(Error,2);



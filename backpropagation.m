function [ delta_w ] = backpropagation( Out_nl, Out,Ih_cell,Ih_nl_cell, delta_w, weights, ...
    current_class, param, current_input)

delta_wih = delta_w{1};
delta_whh = delta_w{2};
delta_woh  = delta_w{3};

hiddenLayersNum = size(weights{2},2)+1;
hiddenUnitsNum = size(delta_wih,3);
if(strcmp(param.nonLinearityFunc, 'sigmoid') == 1)
    f = nonLinearity(param.nonLinearityFunc, Out_nl,'derivative');
else
    f = nonLinearity(param.nonLinearityFunc, Out,'derivative');
end

ErrorDerivative = current_class - Out_nl;
sensitivity = ErrorDerivative.*f; % Delta Rule

Outh = [Ih_nl_cell{hiddenLayersNum} ones(param.batchSize,1)];
for i = 1 : hiddenUnitsNum+1
    delta_woh(:,:,i) = delta_woh(:,:,i)*(1 + param.momentum) + param.etta*sensitivity.*repmat(Outh(:,i),1,size(sensitivity,2));
end

% Update weights (bias weights AS WELL)
if(strcmp(param.nonLinearityFunc, 'sigmoid') == 1)
    f1 = nonLinearity(param.nonLinearityFunc, Ih_nl_cell{hiddenLayersNum}, 'derivative');
else
    f1 = nonLinearity(param.nonLinearityFunc, Ih_cell{hiddenLayersNum}, 'derivative');
end
sensitivity_out = (sensitivity*weights{3}(:,1:end-1)).*f1; % The weights of the bias should NOT go further back

if hiddenLayersNum > 1
    for i = (hiddenLayersNum-1):-1:1
        for j = 1 : hiddenUnitsNum
            delta_whh{i}(:,:,j) = delta_whh{i}(:,:,j)*(1 + param.momentum) + param.etta*repmat(sensitivity_out(:,j),1,hiddenUnitsNum+1) .* [Ih_nl_cell{i} ones(param.batchSize,param.bias)] ; % FIX THIS
            sensitivity = sensitivity_out;
            
            if(strcmp(param.nonLinearityFunc, 'sigmoid') == 1)
                f1 = nonLinearity(param.nonLinearityFunc, Ih_nl_cell{i}, 'derivative');
            else
                f1 = nonLinearity(param.nonLinearityFunc, Ih_cell{i}, 'derivative');
            end
            sensitivity_out = ((weights{2}{i}(1:end-1,:)*sensitivity')').*f1;
        end
    end
end

for j = 1 : hiddenUnitsNum
    delta_wih(:,:,j) = delta_wih(:,:,j)*(1 + param.momentum) + param.etta*repmat(sensitivity_out(:,j),1,size(delta_wih,2)).*current_input;
end
delta_w = {delta_wih,delta_whh,delta_woh};


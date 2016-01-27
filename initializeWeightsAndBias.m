function [wih, woh, whh] = initializeWeightsAndBias(inputUnitsNum, HiddenUnitsNum, HiddenLayersNum, OutputUnitsNum)
    w2 = 1/sqrt(inputUnitsNum + 1); % This is the range the initial weights should be chosen. 
    w1 = -1/sqrt(inputUnitsNum + 1); % They come from a uniform random distribution.
    wih = (w2 - w1).*rand(inputUnitsNum+1,HiddenUnitsNum) + w1; % + 1 because of the bias

    w2h = 1/sqrt(HiddenUnitsNum + 1); % Initial values for the hidden-to-output 
    w1h = -1/sqrt(HiddenUnitsNum + 1); % layer.
    woh = (w2h - w1h).*rand(OutputUnitsNum,HiddenUnitsNum+1) + w1h; % + 1 because of the bias at the hidden layer
    % For more than one hidden layers
    whh = cell(1,HiddenLayersNum - 1);
    if HiddenLayersNum > 1
        for i = 1 : HiddenLayersNum - 1
            whh{i} = (w2h - w1h).*rand(HiddenUnitsNum+1,HiddenUnitsNum) + w1h; 
            % weights for connections between layers + 1 because of the bias
        end
    end

end


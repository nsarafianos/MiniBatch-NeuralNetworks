clear 
clc
close all

% At the beginning in order to train and test every time 
% with the same subset of the dataset save the Data and Classes to a file and load that one

load('fisheriris.mat')
a = 1:size(species,1);
a_rand = a(randperm(length(a)));
RandData = meas(a_rand,:);
RandClasses = species(a_rand,:);

% load('RandDataMat.mat') Uncomment this line and comment the above 5

[class, normalizedData, NumberOfFeatures] = initializeData(RandData,RandClasses);
% set initial parameters.
param.etta=0.01;
param.momentum=0.5;
param.epochs = 1000;
param.batchSize = 30;
param.nonLinearityFunc = 'sigmoid';
param.bias = 1;
trainingSize = size(normalizedData,1)*0.8;
testingSize = size(normalizedData,1)*0.2;
inputUnitsNum = NumberOfFeatures;
outputUnitsNum = size(class,2);

% Select the structure of the Neural Network
hiddenLayersNum = 1;
hiddenUnitsNum = 2;

[wih, woh, whh] = initializeWeightsAndBias(inputUnitsNum, hiddenUnitsNum, hiddenLayersNum, outputUnitsNum);
weights = {wih,whh,woh};

Training_errorPerEpoch = zeros(param.epochs,1);
Test_errorPerEpoch = zeros(param.epochs,1);
TrainingSetNorm = [ normalizedData(1:trainingSize,:) ones(trainingSize,param.bias)];
training_class = class(1:trainingSize,:);
TestingSetNorm = [normalizedData(trainingSize+1:end,1:inputUnitsNum) ones(testingSize,param.bias)];
testing_class = class(trainingSize+1:end,:);

for e = 1:param.epochs
    % initialize delta weights
    delta_woh = zeros(param.batchSize,outputUnitsNum,hiddenUnitsNum+1);
    delta_wih = zeros(param.batchSize,inputUnitsNum+1,hiddenUnitsNum);
    
    if hiddenLayersNum == 1
        delta_whh = zeros(param.batchSize,hiddenUnitsNum+1,hiddenUnitsNum);
    else
        for i = 1 : hiddenLayersNum - 1
            delta_whh{i} = zeros(param.batchSize,hiddenUnitsNum+1,hiddenUnitsNum);
        end
    end
    delta_w = {delta_wih,delta_whh,delta_woh};
    for b = 1:(trainingSize/param.batchSize)
        % setup input for forward pass
        start = (b-1)*param.batchSize+1;
        finish = param.batchSize*b;
        current_input = [normalizedData(start:finish,1:inputUnitsNum),ones(param.batchSize,param.bias)]; % Add ones because of the bias in the input layer
        current_class = class(start:finish,:);

        [Out_nl, Out, Ih_cell,Ih_nl_cell, ~] = forwardPass(current_input, weights, current_class, param);
          
        
        delta_w = backpropagation(Out_nl, Out, Ih_cell,Ih_nl_cell, delta_w, weights, current_class, ...
            param, current_input);
        
        weights = weightUpdate(weights,delta_w);
    end
    
    [~, ~, ~,~, TrainingError] = forwardPass(TrainingSetNorm, weights, training_class, param);
    Training_errorPerEpoch(e) = mean(TrainingError);

    [~, ~, ~, ~, TestingError] = forwardPass(TestingSetNorm, weights, testing_class, param); 
    Test_errorPerEpoch(e) = mean(TestingError);
end


fig = figure('Visible', 'on');
plot(1:param.epochs, Training_errorPerEpoch,1:param.epochs, Test_errorPerEpoch);
title('Error plot');
xlabel('iterations');
ylabel('Error');
legend('training', 'testing');
function [weights] = weightUpdate(weights,delta_w)
%Update Input Weights
SumIn = sum(delta_w{1},1);

for i = 1 : size(weights{1},2)
    weights{1}(:,i) = weights{1}(:,i) + SumIn(1,:,i)';
end
hiddenLayersNum = size(weights{2},2)+1;
%Update Hidden Layer Weights
if hiddenLayersNum > 1
    for i = 1 : hiddenLayersNum- 1
        SumHH = sum(delta_w{2}{i},1);
        for j = 1 : size(weights{2}{i},2)
            weights{2}{i}(:,j) = weights{2}{i}(:,j) + SumHH(1,:,j)';
        end
    end
end

%Update Output Layer Weights
SumOut = sum(delta_w{3},1);
for i = 1 : size(weights{3},2)
    weights{3}(:,i) = weights{3}(:,i) + SumOut(1,:,i)';
end
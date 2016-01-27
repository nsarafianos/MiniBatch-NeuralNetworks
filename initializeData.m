function [class, Norm_data, FeatureNumber] = initializeData(RandData,RandClasses)
    
    [SampleSize,FeatureNumber]=size(RandData);
    %  Standard Score normalization
    meanValues = mean(RandData,1);
    stdValues = std(RandData,1);
    Norm_data = (RandData - repmat(meanValues,SampleSize,1))./repmat(stdValues,SampleSize,1);
    
    class = zeros(SampleSize,size(unique(RandClasses),1));
    for i = 1:size(RandClasses,1)
        if strcmp(RandClasses{i},'setosa') == 1
            class(i,1) = 1;
        elseif strcmp(RandClasses{i},'versicolor') == 1
            class(i,2) = 1;
        else
            class(i,3) = 1;
        end
    end
end

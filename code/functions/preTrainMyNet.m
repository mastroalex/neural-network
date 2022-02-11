function [options,lgraph,dsTrain]=preTrainMyNet(imds, pxds,imageSize,numClasses)

dsTrain = combine(imds, pxds);
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50"); 
% balance predominance of 0
tbl = countEachLabel(pxds);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
options = trainingOptions('sgdm', ...
    'MaxEpochs',50, ...  
    'MiniBatchSize',4, ...
    'Plots','training-progress');
disp('Net ready fro Training')
end
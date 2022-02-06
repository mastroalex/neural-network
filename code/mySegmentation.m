% My segmentation

%% net training
% folder FRAME_TRAIN and GT_TRAIN (i nomi sono scelti dall'utente)
clear all; close all; 

% addpath(strcat(pwd,'/dataset')); %set path 

% set current file path 
filePath = matlab.desktop.editor.getActiveFilename;
pathDivided=strsplit(filePath,'/');
newPath=erase(filePath,pathDivided(end));
% set folder with net syntax
imds = imageDatastore(strcat(newPath,'dataset/FRAME_TRAIN')); % link for training frame
pxds = pixelLabelDatastore(strcat(newPath,'dataset/GT_TRAIN'),["N","B"],[0 1]); % link for GT images

%% 
% RES-NET50
numClasses=2; % problema binario ma si può impostare un problema a più classi
imageSize=[256 256 3]; % dipende dalla rete che si usa
% augmenter serve per ampliare il dataset di immagini e rispettivo GT con
% modifiche rigide quali traslazioni, riflessioni
augmenter = imageDataAugmenter('RandRotation',[0 360],'RandXReflection',true,'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
pximds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter,'OutputSize',imageSize,'ColorPreprocessing','gray2rgb');


lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50"); 
% queste blocco serve per bilanciare rispetto alle classi background e
% foreground
tbl = countEachLabel(pximds);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);


options = trainingOptions('sgdm', ...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Plots','training-progress');

%% Train the network.
% Train the network.
% devo prevedere un resize dei crop per poterli dare in pasto alla rete

net = trainNetwork(pximds,lgraph,options);
% save('Net_CAF_25012020.mat','net','lgraph');
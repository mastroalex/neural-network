% My segmentation

%% net training
% folder FRAME_TRAIN and GT_TRAIN (i nomi sono scelti dall'utente)
clear all; close all; 

% set current file path 
filePath = matlab.desktop.editor.getActiveFilename;
pathDivided=strsplit(filePath,'\');
newPath=erase(filePath,pathDivided(end));
dataPath=strcat(newPath,'dataset');

addpath(strcat(newPath,'functions')); %set path for functions

% set folder with net syntax
imds = imageDatastore(strcat(newPath,'dataset\FRAME_TRAIN')); % link for training frame
pxds = pixelLabelDatastore(strcat(newPath,'dataset\GT_TRAIN'),["N","B"],[0 1]); % link for GT images

%% 
% RES-NET50
net=resnet50;
numClasses=2; % foreground and background
imageSize=net.Layers(1).InputSize; %read size directly from net

augmenter = imageDataAugmenter('RandRotation',[0 360],'RandXReflection',true,'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
pximds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter,'OutputSize',imageSize,'ColorPreprocessing','gray2rgb');

lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50"); 
% balance predominance of 0
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
[net, info]= trainNetwork(pximds,lgraph,options);
save('myNet.mat','net','lgraph');
subfigure(2,1,1);
plot(1:length(info.TrainingAccuracy),info.TrainingAccuracy)
title('Training Accuracy')
subfigure(2,1,2);
plot(info.TrainingLoss)
title('Training Loss')

%% test 
f_test=dir(strcat(dataPath,'/FRAME_TEST_SEG/*.tiff'));
gt_train=dir(strcat(dataPath,'/GT_TEST/*.tiff'));
for l = 1:length(f_test)
    l
testImage=imread([strcat(dataPath,'/FRAME_TEST_SEG/'),f_test(l).name]);
C_test = semanticseg(testImage,net);
D=C_test=='B';
GTImage=imread([strcat(dataPath,'/GT_TEST/'),gt_train(l).name]);
[TP,FP,FN,CR,CM,FM_test(l)]=evaluation_segmentation(bwareafilt(D,1),GTImage);
% imwrite(D,['FRAME_TEST_SEG_NEW\',f_test(l).name(1:end-5),'_seg.tiff']);
imshowpair(testImage,bwareafilt(D,1),'montage');
pause(0.5); drawnow;
clear C_test D testImage;
end
figure;
plot(FM_test);
ylim([0 1]); title('FM')
line([0 length(FM_test)],[mean(FM_test) mean(FM_test)],'Color','red','LineStyle','--');
med=mean(FM_test)
best=max(FM_test)
worst=min(FM_test)
sigma=std(FM_test)

%% cycle over 
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
accuracy=info.TrainingAccuracy(end);
loss=info.TrainingLoss(end);

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
FM_test_sgdm=FM_test;
med=mean(FM_test)
best=max(FM_test)
worst=min(FM_test)
sigma=std(FM_test)

%% Solver
options_rmsprop = trainingOptions('rmsprop', ...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Plots','training-progress');
[net_rmsprop, info]=trainNetwork(pximds,lgraph,options_rmsprop);
accuracy_rmsprop=info.TrainingAccuracy(end);
loss_rmsprop=info.TrainingLoss(end);

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
FM_test_rmsprop=FM_test;
med_rmsprop=mean(FM_test)
best_rmsprop=max(FM_test)
worst_rmsprop=min(FM_test)
sigma_rmsprop=std(FM_test)

options_adam = trainingOptions('adam', ...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Plots','training-progress');
[net_adam, info]=trainNetwork(pximds,lgraph,options_adam);
accuracy_adam=info.TrainingAccuracy(end);
loss_adam=info.TrainingLoss(end);

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
FM_test_adam=FM_test;
med_adam=mean(FM_test)
best_adam=max(FM_test)
worst_adam=min(FM_test)
sigma_adam=std(FM_test)

figure;
plot(FM_test_sgdm,'r');
hold on
plot(FM_test_adam,'g');
plot(FM_test_rmsprop,'b');
line([0 length(FM_test_sgdm)],[mean(FM_test_sgdm) mean(FM_test_sgdm)],'Color','red','LineStyle','--');
line([0 length(FM_test_adam)],[mean(FM_test_adam) mean(FM_test_adam)],'Color','green','LineStyle','--');
line([0 length(FM_test_rmsprop)],[mean(FM_test_rmsprop) mean(FM_test_rmsprop)],'Color','blue','LineStyle','--');
hold off
legend({'sdgm','adam','rmsprop',string(mean(FM_test_sgdm)),string(mean(FM_test_adam)),string(mean(FM_test_rmsprop))})

%% Epoche 

epoc=[5 10 20 30 40 50 60];
for i=1:length(epoc)
   disp(strcat('epoche:'," ",string(epoc(i))));
    options = trainingOptions('sgdm', ...
    'MaxEpochs',epoc(i), ...  
    'MiniBatchSize',8, ...
    'Plots','training-progress');
[net_epoc, info]= trainNetwork(pximds,lgraph,options);
accuracy_epoc(i)=info.TrainingAccuracy(end);
loss_epoc(i)=info.TrainingLoss(end);

for l = 1:length(f_test)
testImage=imread([strcat(dataPath,'/FRAME_TEST_SEG/'),f_test(l).name]);
C_test = semanticseg(testImage,net);
D=C_test=='B';
GTImage=imread([strcat(dataPath,'/GT_TEST/'),gt_train(l).name]);
[TP,FP,FN,CR,CM,FM_test(l)]=evaluation_segmentation(bwareafilt(D,1),GTImage);
clear C_test D testImage;
end
FM_test_epoc(i,:)=FM_test;
end


% X boxplot usare figure;boxplot([FM_test', FM_test_adam', FM_test_rmsprop', FM_test_sgdm'])



function [net, info, FM_test,compTime]=trainAndTest(pximds,lgraph,dataPath,f_test,gt_train)

options = trainingOptions('sgdm', ...
    'MaxEpochs',40, ...  
    'MiniBatchSize',8, ...
    'Plots','training-progress');
%Training 
[net, info]= trainNetwork(pximds,lgraph,options);

%Test
tic
for l = 1:length(f_test)
testImage=imread([strcat(dataPath,'/FRAME_TEST_SEG/'),f_test(l).name]);
C_test = semanticseg(testImage,net);
D=C_test=='B';
GTImage=imread([strcat(dataPath,'/GT_TEST/'),gt_train(l).name]);
[TP,FP,FN,CR,CM,FM_test(l)]=evaluation_segmentation(bwareafilt(D,1),GTImage);
clear C_test D testImage;
end
compTime=toc;

end

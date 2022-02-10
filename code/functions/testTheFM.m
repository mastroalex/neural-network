function [FM_test]=testTheFM(f_test,gt_train,net,dataPath,l)

testImage=imread([strcat(dataPath,'/FRAME_TEST_SEG/'),f_test(l).name]);
C_test = semanticseg(testImage,net);
D=C_test=='B';
GTImage=imread([strcat(dataPath,'/GT_TEST/'),gt_train(l).name]);
[TP,FP,FN,CR,CM,FM_test]=evaluation_segmentation(bwareafilt(D,1),GTImage);

end
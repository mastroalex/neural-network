function [accuracy,loss,med]=figureAccAndLoss(info,FM_test)
subfigure(2,1,1);
plot(1:length(info.TrainingAccuracy),info.TrainingAccuracy,'Color','#0072BD')
title('Training Accuracy')
subfigure(2,1,2);
plot(info.TrainingLoss,'Color','#D95319')
title('Training Loss')
accuracy=info.TrainingAccuracy(end);
loss=info.TrainingLoss(end);

figure;
plot(FM_test);
ylim([0 1]); title('FM')
line([0 length(FM_test)],[mean(FM_test) mean(FM_test)],'Color','red','LineStyle','--');
med=mean(FM_test);
best=max(FM_test);
worst=min(FM_test);
sigma=std(FM_test);
disp(strcat('FM mean:'," ",string(med)))
disp(strcat('FM max:'," ",string(best)))
disp(strcat('FM min:'," ",string(worst)))
disp(strcat('FM std:'," ",string(sigma)))
disp(strcat('accuracy:'," ",string(accuracy)))
disp(strcat('loss:'," ",string(loss)))
end
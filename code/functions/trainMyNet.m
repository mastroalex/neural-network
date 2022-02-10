function [net, accuracy, loss_batch]=trainMyNet(pximds,lgraph,options)
[net, info]= trainNetwork(pximds,lgraph,options);
accuracy=info.TrainingAccuracy(end);
loss_batch=info.TrainingLoss(end);

end
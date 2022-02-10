function [pximds,lgraph]=prepareMyNet(Mynet,Netname,imds,pxds)
net=Mynet;
numClasses=2; % foreground and background
imageSize=net.Layers(1).InputSize; %read size directly from net
augmenter = imageDataAugmenter('RandRotation',[0 360],'RandXReflection',true,'RandXTranslation',[-20 20],'RandYTranslation',[-20 20]);
pximds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter,'OutputSize',imageSize,'ColorPreprocessing','gray2rgb');
lgraph = deeplabv3plusLayers(imageSize, numClasses, Netname); 
% balance predominance of 0
tbl = countEachLabel(pximds);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
disp(strcat(Netname," ",'ready'))

end
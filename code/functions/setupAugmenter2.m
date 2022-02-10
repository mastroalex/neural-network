function [lgraph,pximds,options]=setupAugmenter2(imds,pxds,imageSize,numClasses,value,value1,TypeOfAnalysis,TypeOfAugmenter)

  arguments
      imds
      pxds
      imageSize
      numClasses
      value
      value1
           TypeOfAnalysis (1,:) char {mustBeMember(TypeOfAnalysis,{'full','partial'})} = 'full'
        TypeOfAugmenter (1,:) char {mustBeMember(TypeOfAugmenter,{'rotation','reflection','traslation'})} = 'rotation'
  end

 if  strcmp(TypeOfAnalysis,'full')
     
     if strcmp(TypeOfAugmenter,'rotation')
        augmenter = imageDataAugmenter('RandRotation',value);       
     end
       if strcmp(TypeOfAugmenter,'reflection')
        augmenter = imageDataAugmenter('RandXReflection',true);       
       end
       if strcmp(TypeOfAugmenter,'traslation')
        augmenter = imageDataAugmenter('RandXTranslation',value,'RandYTranslation',value);       
       end     
 end
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 if strcmp(TypeOfAnalysis,'partial')
     if strcmp(TypeOfAugmenter,'rotation')
        augmenter = imageDataAugmenter('RandRotation',value,'RandXReflection',true,'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);       
     end
       if strcmp(TypeOfAugmenter,'reflection')
        augmenter = imageDataAugmenter('RandRotation',[0 360],'RandXReflection',true,'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);       
       end
       if strcmp(TypeOfAugmenter,'traslation')
        augmenter = imageDataAugmenter('RandRotation',[0 360],'RandXReflection',true,'RandXTranslation',value,'RandYTranslation',value1);       
       end      
 end
    
 pximds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter,'OutputSize',imageSize,'ColorPreprocessing','gray2rgb');
        lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50"); 
        % balance predominance of 0
        tbl = countEachLabel(pximds);
        totalNumberOfPixels = sum(tbl.PixelCount);
        frequency = tbl.PixelCount / totalNumberOfPixels;
        classWeights = 1./frequency;
 pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
% SET EPOchs and BAYCH
options = trainingOptions('sgdm', ...
    'MaxEpochs',20, ...  
    'MiniBatchSize',8, ...
    'InitialLearnRate',0.001, ... 
    'Plots','training-progress');
 

end



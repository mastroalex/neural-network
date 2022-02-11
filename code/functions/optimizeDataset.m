function [imds,pxds]=optimizeDataset(pathImage,pathGT,pathExport,elementSize,imageSize)

imds=imageDatastore(pathImage);
startingGT=imageDatastore(pathGT);
SE = strel('diamond',elementSize);
%SE = strel('square',5)
disp('erosion element:')
figure;
imshow(SE.Neighborhood)
% resize and prepare al GT image for segmentation
f = waitbar(0,'Please wait...');
for i=1:length(startingGT.Files)
    %disp(strcat('current:',string(i)," ",'of'," ",string(length(startingGT.Files))))
    waitbar(i/length(startingGT.Files),f,strcat('Loading your data:'," ",string(i)," of ",string(length(startingGT.Files))));
    clear GT temp_GT temp_INT 
    temp_GT=imread(string(startingGT.Files(i)));
    %temp_INT=bwperim(test);
    temp_INT=imerode(temp_GT,SE); % delete border
    temp_INT=imbinarize(temp_INT);  %clear numeric error
    temp_GT=imbinarize(temp_GT);
    temp_GT=uint8(temp_GT); %conversion from logic 
    temp_INT=uint8(temp_INT); %conversion from logic
    imshowpair(255*temp_GT,255*temp_INT)
    GT=temp_GT+temp_INT; % 2 on internal, 1 on border 
    %GT=cat(3, GT,GT,GT); % convert into RGB (for training with RGB)
    drawnow;
    GT=imresize(GT,imageSize(1:2));
    %pause(0.5);
    pathSplit=strsplit(string(startingGT.Files(i)),'\');
    imwrite(GT,strcat(pathExport,'GT\',pathSplit(end)))
end


for i=1:length(imds.Files)
    imm=imread(string(imds.Files(i)));
    imm=imresize(imm,imageSize(1:2));
    pathSplit=strsplit(string(imds.Files(i)),'\');
    imwrite(imm,strcat(pathExport,'IMG\',pathSplit(end)))
end

pxds = pixelLabelDatastore(strcat(pathExport,'GT'),["BACK","BORDER","INSIDE"],[0 1 2]); % link for GT images
imds = imageDatastore(strcat(pathExport,'IMG'));
end
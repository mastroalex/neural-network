function [area]=testBorderSegmentation(imdsTEST,index,net,mode)
testImage=imread(string(imdsTEST.Files(index)));
C_test = semanticseg(testImage,net);

D=C_test=='INSIDE';
E=C_test=='BORDER';
F=C_test=='BACK';
if strcmp(mode,'verbose')
disp('inside:')
figure;
imshow(D)
disp('border:')
figure;
imshow(E)
disp('back:')
figure;
imshow(F)
end

% Post processing
D=imfill(D,'holes');
G=E-D;
G(G<0)=0;
SE2 = strel('disk',15);
G=imclose(G,SE2);
imm=readimage(imdsTEST,index);
figure;
p=imshowpair(imm,D,'falsecolor');
figure;
imshowpair(imm,p.CData,'montage')
figure;
figure;
p1=imshowpair(imm,G,'falsecolor');
figure;
imshowpair(imm,p1.CData,'montage')
area=sum(sum(D))/numel(D);
disp(strcat('internal area:'," ",string(area*100)," ",'%'));
figure;
imshowpair(D,G,'montage')

end
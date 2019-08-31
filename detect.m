%%Image segmentation and extraction

imagen=imread('C:\Users\deepa\Downloads\char.jpg');
figure,imshow(imagen);
title('INPUT IMAGE WITH NOISE')

imagen=rgb2gray(imagen);

% Convert to binary image
threshold = graythresh(imagen);
imagen = ~imbinarize(imagen,threshold);

%Remove all object containing fewer than 30 pixels
imagen = bwareaopen(imagen,30);
figure, imshow(~imagen);
title('INPUT IMAGE WITHOUT NOISE')

%Label connected components
[L, Ne]=bwlabel(imagen);

%Measure properties of image regions
propied=regionprops(L,'BoundingBox');

hold on
%%Plot Bounding Box
for n=1:size(propied,1)
  rectangle('Position',propied(n).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off

% Objects extraction
for n=1:Ne
  [u, v] = find(L==n);
  n1=imagen(min(u):max(u),min(v):max(v));
  figure, imshow(~n1);
end
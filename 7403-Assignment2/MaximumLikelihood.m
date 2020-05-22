clc
clear all;
close all;
%%1.use the training data to find mean and variance of two classes
for i = 35:40
    j = i - 34;
    x = sprintf('%d',i);
    pimg{j} = [x,'_training.tif'];%training data
    y{j} = [x,'_manual2.gif'];%label
    [mu0(j),sigma0(j),mu1(j),sigma1(j)] = train(pimg{j},y{j});%mean and variance of each training image
end
 
%%2.find mean and variance over all training data
MU0 = mean(mu0);
Sigma0 = mean(sigma0);%overall mean and variance of class 0
MU1 = mean(mu1);
Sigma1 = mean(sigma1);%overall mean and variance of class 1
 
%%3.use the test image to classify each pixel
img=imread('24_training.tif');
img=img(:,:,2);%the green component
x=img;
m=graythresh(x)*255; %calculated threshold based on otsu algorithm
x(find(x<m)) = 0;
x(find(x>=m)) = 1;%binarization for plot the contour
figure(2); 
imshow(x,[]);
title('contour of the test image')
pimg = img.*x;%remove contour
sq = strel('square',8);
pimg = imbothat(pimg,sq);%bottom-hat transformation
pimg = double(pimg);
figure(3)
imshow(pimg,[]);
title('remove the contour')
[a,b] = size(img);
map = zeros(a,b);
pimg = medfilt2(pimg,[3 3]);
figure(4)
imshow(pimg,[])
title('vessel after filtering')
%maximum likelihood estimation
for i = 1:a
    for j = 1:b
    pr0 = -1 / 2 * log(2 * pi * Sigma0) - (pimg(i,j) - MU0).^2 / (2 * Sigma0);%the probability that this pixel belongs to class 0
    pr1 = -1 / 2 * log(2 * pi * Sigma1) - (pimg(i,j) - MU1).^2 / (2 * Sigma1);%the probability that this pixel belongs to class 1
    if pimg(i,j) ~= 0 && pr1 > pr0 %classified as vessel
       map(i,j) = 255;
    end
    end
end
map = uint8(map);
figure(5);
imshow(map);
title('the final result after ML estimation')
 
%%4.accuracy
gtruth = imread('24_manual1.gif');
[c,d]=size(gtruth);
tst=zeros(c,d);
tst(find(map == gtruth)) = 1;
accuracy = 100*sum(sum(tst))/(c*d);
 
%%5.training function
function [mu0,sigma0,mu1,sigma1]=train(img,label)
Img = imread(img);
Img = Img(:,:,2);
GTruth = imread(label);
[e,f] = size(Img);
 
%segment the image area
x = Img;
m = graythresh(x) * 255; %calculated threshold based on otsu algorithm
x(find(x<m)) = 0;
x(find(x>=m)) = 1;%binarization for plot the contour
figure(1); 
imshow(x,[]);
title('contour of the training image')
img = Img.*x;%remove contour
sq = strel('square',8);%create an 8*8 square
img = imbothat(img,sq);%bottom-hat transform
count0 = 0;
count1 = 0;
img = medfilt2(img,[2 2]); %filtering
for i = 1:e
    for j = 1:f
   %class0 is background
   if img(i,j) ~= 0 && GTruth(i,j) == 0
        count0 = count0+1;
        c0(count0) = img(i,j);
    end
   %class1 is vessel
   if GTruth(i,j) == 255
      count1 = count1+1;
      c1(count1) = img(i,j);
   end
    end
end
%the mean and covariance for class 0 in this image
mu0 = mean(c0);
sigma0 = mean((c0 - mu0).^2);
%the mean and covariance for class 1 in this image
mu1 = mean(c1);
sigma1 = mean((c1 - mu1).^2);
end 
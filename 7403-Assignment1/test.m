clc
clear all;
close all;
image=imread('04_test.tif'); %load the image

%1.choose the green component
imager=image(:,:,1);%red component
imageg=image(:,:,2);%green component
imageb=image(:,:,3);%blue component

figure;
subplot(131);imshow(imager);title('red channel')
subplot(132);imshow(imageg);title('green channel')
subplot(133);imshow(imageb);title('blue channel')

%2.histogram equalization
figure;
imhist(imageg);title('histogram of the green channel')
figure;
subplot(121);
[row,col]=size(imageg);
countg=zeros(1,256);%counter for every pixel value
for i=1:row
    for j=1:col
        countg(1,imageg(i,j)+1)=countg(1,imageg(i,j)+1)+1;
    end
end
T=zeros(1,256);%define the transform
T=double(T);countg=double(countg);
%count the probability of occurrence of each pixel value
for i=1:256
    T(1,i)=countg(1,i)/(row*col);
end
%accumulated probability distribution
for i=2:256
    T(1,i)=T(1,i-1)+T(1,i);
end
for i=1:256
    T(1,i)=T(1,i)*255;
end
H1=double(imageg);
for i=1:row
    for j=1:col
        H1(i,j)=T(1,H1(i,j)+1);
    end
end
H1=uint8(H1);
imshow(H1);title('figure after histogram equalization');
subplot(122);
imhist(H1);title('histogram after histogram equalization');

%3.Match filter
sigma=2; %change the thickness of vessels
yLength=14; %the larger L, the more obvious the smoothing effect
direction_number=21;
mf=MatchFilter(H1,sigma,yLength,direction_number);%match filtering
mask=[0 0 0 0 0;
    0 1 1 1 0;
    0 1 1 1 0;
    0 1 1 1 0;
    0 0 0 0 0;];
mf(mask==0)=0;
eff_img = mf;      %Effect picture of matched filtering
figure;
subplot(121);
imshow(eff_img*3);
title('Matched Filtering');

%Binarization
threshold2 = graythresh(eff_img)
binary_data1 = imbinarize(eff_img,threshold2);%Binarize the image
subplot(122);
imshow(binary_data1);
title('binary graph');

%% 4.remove contour
img = imread('24_training.tif');
img = img(:,:,2);%green component
[m n] = size(img);
hist_counter = zeros(1,256);
for i = 1:m
    for j = 1:n
        hist_counter(img(i, j)+1) = hist_counter(img(i, j)+1) + 1;
    end
end
hist_pro = hist_counter/m/n;         %gray level probability density distribution
sigma2_max = 0;
theta = 0;
for t = 0:255%otsu algorithm
    w0 = 0;
    w1 = 0; 
    u0 = 0;
    u1 = 0; 
    u = 0;
    for q = 0:255 %each possibible gray-scale pixel value
        if q <= t
            w0 = w0 + hist_pro(q+1);
            u0 = u0 + (q)*hist_pro(q+1);
        else
            w1 = w1 + hist_pro(q+1);
            u1 = u1 + (q)*hist_pro(q+1);
        end
    end
    u = u0 + u1;
    u0 = u0 / (w0+eps);
    u1 = u1 / (w1+eps);
    sigma2 = w0 * (u0 - u)^2 + w1 * (u1 - u)^2;     %find the maximum between classes
    if (sigma2 > sigma2_max)
        sigma2_max = sigma2;
        theta = t;
    end
end
img_out = img;
for i = 1:m                                         %thresholding for contour
    for j = 1:n
        if img(i, j) >= theta;
            img_out(i, j) = 255;
        else 
            img_out(i, j) = 0;
        end
    end
end

figure;
subplot(221);
imshow(img_out);
title('mask');

new_img = imsubtract(double(eff_img),double(img_out));
subplot(222);
imshow(new_img);
title('Contour of grayscale');

theta = graythresh(new_img)
SE=strel('disk',6);      %mask
new_img=imdilate(new_img,SE); %dilation
subplot(223);
binary_data2 = imbinarize(new_img,theta);%binaryize the image and manually adjust
imshow(binary_data2);
title('Contour binary map');

binary_data = binary_data1-binary_data2;
subplot(224);
imshow(binary_data);
title('The final binary map');

%% 5.image processing:removing noise points; continuous breakpoints; corrosion expansion, etc.
sigma1=10;
gFilter=fspecial('gaussian',[5 5],sigma1);
result = imfilter(binary_data,gFilter,'replicate');
figure;
imshow(result);
title('Gaussian filtering');

%l1=imread('24_manual1.gif');
%l2=result;
%[h,w]=size(l2);
%tst=zeros(h,w);
%tst(find(uint8(l1)==uint8(l2)))=1;
%accuracy=100*sum(sum(tst))/(h*w)
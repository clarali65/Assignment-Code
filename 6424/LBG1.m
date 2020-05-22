clear
clc
codebook=[-2,2;-1,2];%Initial codebook
tdata=[-2,2;-1,2;-1,1;0,0;3,0;1,-2];%training data
m1=[];%region 1
m2=[];%region 2
for i=1:6
    if sum((tdata(i,:)-codebook(1,:)).^2,2)<sum((tdata(i,:)-codebook(2,:)).^2,2)%Euclidean distance
        m1=[m1;tdata(i,:)];
    else
        m2=[m2;tdata(i,:)];
    end
end
codebook(1,:)=mean(m1,1);%codebook update
codebook(2,:)=mean(m2,1);%codebook update
codebook
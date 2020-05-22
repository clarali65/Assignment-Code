clear
clc
t=0.3;%predefined threshold
iter=1;
flag=true;
codebook=[-2,2;-1,2;12,-2];%Initial codebook
tdata=[-2,2;-1,2;-1,1;0,0;3,0;1,-2;0,2;4,0;4,-1;4,-3;7,-3;6,-5;9,1;10,5;12,-1];%training data
while flag
    m1=[];%region1
    m2=[];%region2
    m3=[];%region3
    for i=1:15
        error=0;%total error in this iteration
        errorj=[];  
        for j=1:3
            errorj=[errorj;sum((tdata(i,:)-codebook(j,:)).^2,2)];%Euclidean distance
        end
        [~,idx]=min(errorj); %find the minimum euclidean distance
        if idx==1%find the corresponding region
            m1=[m1;tdata(i,:)];
        elseif idx==2
            m2=[m2;tdata(i,:)];
        else
            m3=[m3;tdata(i,:)];
        end
    end
    codebook(1,:)=mean(m1,1);%codebook update
    codebook(2,:)=mean(m2,1);%codebook update
    codebook(3,:)=mean(m3,1);%codebook update
    [a,b]=size(m1);
    [c,d]=size(m2);
    [e,f]=size(m3);
    %the total error
    for k=1:a
        error=error+sum((m1(k,:)-codebook(1,:)).^2,2);
    end
    for k=1:c
        error=error+sum((m2(k,:)-codebook(2,:)).^2,2);
    end
    for k=1:e
        error=error+sum((m3(k,:)-codebook(3,:)).^2,2);
    end
    codebook
    fprintf('iteration:%d\n',iter)
    fprintf('error:%f\n',error)
    if error<t
        flag=false;
    end
    iter=iter+1
end
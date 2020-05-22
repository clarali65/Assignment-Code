clc;
clear all;
close all;

Music_files = {'test.wav'}; % Audio files

fs = 16000;
for i=1:1  
    frame_num=228;
    
    [y{i}, Fs{i}] = audioread(Music_files{i}); % Reading the files
    %% Step 1
    r1 = splnorm(y{i},Fs{i}); % SPL normalization
    %% Step 2
    [tonalm,tonal_bark,noise,noise_bark,tonal_bin,noise_bin] = tonal(r1,Fs{i},frame_num); % Find thte tonal and non-tonal component
    %% Step 3
    [tonalm_t,tonal_bark_t,noise_t,noise_bark_t,tonal_dec, tbark_dec, noise_dec, nbark_dec, t_bin, n_bin] = thresh(tonalm,tonal_bark,noise,noise_bark,tonal_bin,noise_bin, Fs{i}); % Decimation and reorganization of maskers
    %% Step 4
    [T_TM, T_NM, t_bark, n_bark,t,n] = indmask(tonal_dec,noise_dec,t_bin, n_bin, r1(1:256,frame_num)); % Find the individual tonal and non-tonal mask threshold
    %% Step 5
    [TG, nfrewdrop] = globmaskthresh(T_TM,T_NM,r1(1:256,frame_num),t,n, t_bark,n_bark);% Find the global mask threshold
    %% Plotting 
    f=1:256;
    f_bark = freq2bark(f,0);
    %Tq2 = real(10*log10(tq(f)));
    Tq = tq(f); % Absolute Threhold for Quiet
    figure;
    plot(f_bark,r1(1:256,frame_num));
    hold on;
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step1: PSD- SPL Normalized')
    hold on;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    ylim([-50 150]);
    legend('Original Signal','Quiet Threshold')
    
    hold off;
 
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num));
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step2 :Tonal+ Noise Maskers')
    hold on;
    plot(tonal_bark,tonalm,'x','LineWidth',2,'DisplayName','Tonal masker');
    hold on;
    plot(noise_bark,noise,'o','LineWidth',2,'DisplayName','Noise masker');
    yLimits = get(gca,'YLim');  %# Get the range of the y axis
    for j=1:length(tonal_bark)
        line([tonal_bark(j) tonal_bark(j)],[tonalm(j) yLimits(1)],'linestyle',':','Color','r');
    end
    for j=1:length(noise_bark)
        line([noise_bark(j) noise_bark(j)],[noise(j) yLimits(1)],'linestyle',':','Color','g');
    end
    ylim([-50 150]);
    legend('Quiet Threshold','Original Signal','Tonal masker','Noise Masker')
    
    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num));
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step3 :Tonal,Noise Maskers + Threshold + Decimation')
    hold on;
    plot(tbark_dec,tonal_dec,'x','LineWidth',2,'DisplayName','Tonal masker');
    hold on;
    plot(nbark_dec,noise_dec,'o','LineWidth',2,'DisplayName','Noise masker');
    yLimits = get(gca,'YLim');  %# Get the range of the y axis
    for j=1:length(tbark_dec)
        line([tbark_dec(j) tbark_dec(j)],[tonal_dec(j) yLimits(1)],'linestyle',':','Color','r');
    end
    for j=1:length(noise_dec)
        line([nbark_dec(j) nbark_dec(j)],[noise_dec(j) yLimits(1)],'linestyle',':','Color','g');
    end
    ylim([-50 150]);
    legend('Quiet Threshold','Original Signal','Tonal masker','Noise Masker')


    
    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num),'linestyle',':');
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step4 :Tonal Maskers + Indiv. Tonal Maskers')
    hold on;
    plot(t_bark,tonal_dec,'x','LineWidth',2,'DisplayName','Tonal masker');
    hold on;
    plot(f_bark,T_TM);
    ylim([-50 150]); 
        legend('Quiet Threshold','Original Signal','Tonal masker')

    
    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k','linestyle',':');
    hold on;
    plot(f_bark,r1(1:256,frame_num),'linestyle',':');
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step4 :Noise Maskers + Indiv. Noise Maskers')
    hold on;
    plot(n_bark,noise_dec,'o','LineWidth',2,'DisplayName','Noise masker');

    hold on;
    plot(f_bark,T_NM);
    ylim([-50 150]);
        legend('Quiet Threshold','Original Signal','Noise Masker')


    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num),'linestyle',':');
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step5 :Global Masking Threshold + Tonal & Noise Maskers ')
    hold on;
    plot(t_bark,tonal_dec,'x','LineWidth',2,'DisplayName','Noise masker');
    hold on;
    plot(n_bark,noise_dec,'o','LineWidth',2,'DisplayName','Tonal masker');    
    hold on;
    plot(f_bark,TG);
    ylim([-50 150]);
        legend('Quiet Threshold','Original Signal','Tonal masker','Noise Masker','Global Threshold')

end

%%
function snorm = splnorm(X,fs)
s = X(:,1);
n = 512;
b = 16;
preemph = [1 -0.97];
s = filter(1,preemph,s);

x=s;
N=floor(length(x)/512); %find how many samples will each frame contain after applying 512-point FFT
n_overlap_frames = floor((N*512-512)/480); % number of overlap frames
x_overlap_16 = zeros(512,n_overlap_frames); % number of overlap data points
x_hann = zeros(512,n_overlap_frames); % for windowed data
x_fft = zeros(512,n_overlap_frames); % spectral lines
psd = zeros(512,n_overlap_frames);% power density spectrum
P = zeros(512,n_overlap_frames);% power spectrum density after normalization in dB scale
PN=90.302;
for k=0:n_overlap_frames
    x_overlap_16(:,k+1)=x(1+(n*k*15/16):n*(k+1)-((k*n)/16));
    x_hann(:,k+1)= hanning(length(x_overlap_16(:,k+1))).*x_overlap_16(:,k+1);
    x_fft(:,k+1)= fft(x_hann(:,k+1));
    psd(:,k+1)= (abs(x_fft(:,k+1)).^2);
    P(:,k+1)= PN + 10*log10(psd(:,k+1));
end
snorm =  P;

end

%%
function [tone,fre,noise,kbar,tbin,nbin] = tonal(X,fs,frame_num)
len = length(X);
c=1;
S=[];% record the spectral lines of each tonal masker
for k=frame_num:frame_num
    for j=1:256 
        if j>2 && j<63
            if X(j,k)>X(j-1,k) &&  X(j,k)>X(j+1,k) && X(j,k)>(X(j-2,k)+7) && X(j,k)>(X(j+2,k)+7)    
                S = [S;j];
                c=c+1;
            end
        elseif j>62 && j<127
            if X(j,k)>X(j-1,k) &&  X(j,k)>X(j+1,k) && X(j,k)>X(j-3,k)+7 && X(j,k)>X(j+3,k)+7 && X(j,k)>(X(j-2,k)+7) && X(j,k)>(X(j+2,k)+7)  
                 S = [S;j];
                c=c+1;
            end
        elseif j>126 && j<257
             if X(j,k)>X(j-1,k) &&  X(j,k)>X(j+1,k) && X(j,k)>X(j-6,k)+7 && X(j,k)>X(j+6,k)+7 && X(j,k)>X(j-3,k)+7 && X(j,k)>X(j+3,k)+7 && X(j,k)>(X(j-2,k)+7) && X(j,k)>(X(j+2,k)+7)&& X(j,k)>X(j-4,k)+7 && X(j,k)>X(j+4,k)+7 && X(j,k)>(X(j-5,k)+7) && X(j,k)>(X(j+5,k)+7)    
                 S = [S;j];
                c=c+1;
             end
        end
    end
end


[row, col]=size(S);
P_TM= zeros(row,1);
for k=frame_num:frame_num
    for j=1:row
        P_TM(j,1) = 10*log10(10^(0.1*X(S(j)-1,k))+10^(0.1*X(S(j),k))+10^(0.1*X(S(j)+1,k))); % magnitude of tonal masker
    end
end
tone = P_TM;  % magnitude of each tonal masker
tbin = round(S);

tonal_bark = freq2bark(S,0);
fre = tonal_bark; % index of each tonal frequency lines
k_bar = zeros(1,length(S));
for i=1:length(S)
    if i==1 || i==length(S)
        k_bar(i)= S(i);
    else
        k_bar(i) = geomean([S(i-1) S(i+1)]); % the geometric mean spectral line of the critical band
   end
end
kbar2=zeros(25,1);
kbar2(1)= geomean([0 100]);
kbar2(2)= geomean([100 200]);
kbar2(3)= geomean([200 300]);
kbar2(4)= geomean([300 400]);
kbar2(5)= geomean([400 510]);
kbar2(6)= geomean([510 630]);
kbar2(7)= geomean([630 770]);
kbar2(8)= geomean([770 920]);
kbar2(9)= geomean([920 1080]);
kbar2(10)= geomean([1080 1270]);
kbar2(11)= geomean([1270 1480]);
kbar2(12)= geomean([1480 1720]);
kbar2(13)= geomean([1720 2000]);
kbar2(14)= geomean([2000 2320]);
kbar2(15)= geomean([2320 2700]);
kbar2(16)= geomean([2700 3150]);
kbar2(17)= geomean([3150 3700]);
kbar2(18)= geomean([3700 4400]);
kbar2(19)= geomean([4400 5300]);
kbar2(20)= geomean([5300 6400]);
kbar2(21)= geomean([6400 7700]);
kbar2(22)= geomean([7700 9500]);
kbar2(23)= geomean([9500 12000]);
kbar2(24)= geomean([12000 15500]);
kbar2(25)= geomean([15500 22050]);

nbin = round(kbar2/44100*512);
nbin(nbin==0)=1;
kbar3 = zeros(length(S),1);

bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];
P_NM2 = zeros(length(S),1);
d=1;
S2 = (S*44100/512);
for k=frame_num:frame_num
    for j=1:length(bw)-1
        X2=X;
            u = round(bw(j+1)/44100*512);
            if j==1
                l=1;
            else
                l = round(bw(j)/44100*512);
            end
            P_NM2(d,1) = 10*log10(sum(10.^(0.1*X2(l:u,k)))); % magnitude of the non-tonal masker
            d=d+1;
    end
end

d=1;
for k=frame_num:frame_num
    for j=1:length(bw)-1
        X2=X;
        if d>length(S2)
            break;
        end
        if S2(d)<bw(j+1)
           
            kbar3(d)=j;
           
            d=d+1;
            j=1;
        end
    end
end

kbar2(kbar3)=[];
nbin = round(kbar2/44100*512);
nbin(nbin==0)=1;
kbar = freq2bark(nbin,0);

m=0;

P_NM2(kbar3)=[];
noise = P_NM2; % magnitude of each non-tonal masker

end

%% 
function [tonal2, tb, noise2, nb,tonal3, tb2, noise3, nb2, tonal_bin2, noise_bin2 ] = thresh(tonalm,tonal_bark,noise,noise_bark, tonal_bin, noise_bin,fs)
len = length(tonalm);
f=1:256;
Tq = tq(f);% quiet threshold
aa=0;
t=0;
n=0;

for i=1:len
    if tonalm(i)< Tq(tonal_bin(i))% discard tonal maskers
        disp(i);
        tonal_bark(i)=-5555;
        tonal_bin(i)=-5555;
        tonalm(i)=-5555;       
        t=t+1;
    end
end
tonal_bark=tonal_bark(tonal_bark~=-5555);
tonal_bin=tonal_bin(tonal_bin~=-5555);
tonalm=tonalm(tonalm~=-5555);
len= length(tonalm);

for i=1:length(noise_bin)
    if noise(i)< Tq(noise_bin(i)) % discard non-tonal maskers
        noise_bark(i)=-5555;
        noise_bin(i)=-5555;
        noise(i)=-5555;
     
        n=n+1;
    end
end
noise_bark=noise_bark(noise_bark~=-5555);
noise_bin=noise_bin(noise_bin~=-5555);
noise=noise(noise~=-5555);

combo = [tonal_bark, noise_bark];
combo2 = [tonal_bin.', noise_bin.'];
combo3 = [tonalm.', noise.']
[sorted, ind] = sort(combo);
for i=1:length(sorted)-1
    if sorted(i+1)-sorted(i)<=0.5 % stronger masker within a pair of masker will survive
        ind(i)=-5555;
    end
end
ind=ind(ind~=-5555);
ind_t = ind(ind<=length(tonalm));
ind_n = ind(ind>length(tonalm));
tb = combo(ind_t);
nb = combo(ind_n);
tonal_bin = combo2(ind_t);
noise_bin = combo2(ind_n);
tonal2 = combo3(ind_t);
noise2 = combo3(ind_n);
c=1;
i_tonal=[];
i_noise=[];

% reorganize the masker frequency bins
for i=1:length(tonal_bin)
    if tonal_bin(i)>=1 && tonal_bin(i)<=48
        i_tonal(c)=tonal_bin(i);
        c=c+1;
    elseif tonal_bin(i)>=49 && tonal_bin(i)<=96
        i_tonal(c)=tonal_bin(i)+ mod(tonal_bin(i),2);
        c=c+1;
    elseif tonal_bin(i)>=97 && tonal_bin(i)<=232
        i_tonal(c)=tonal_bin(i)+ 3 -( mod(tonal_bin(i)-1,4));
        c=c+1;
    else 
        tonal2(i)=-1;
    end
end
c=1;

for i=1:length(noise_bin)
    if noise_bin(i)>=1 && noise_bin(i)<=48
        i_noise(c)=noise_bin(i);
        c=c+1;
    elseif noise_bin(i)>=49 && noise_bin(i)<=96
        i_noise(c)=noise_bin(i)+ mod(noise_bin(i),2);
        c=c+1;
    elseif noise_bin(i)>=97 && noise_bin(i)<=232
        i_noise(c)=noise_bin(i)+ 3 -( mod(noise_bin(i)-1,4));
        c=c+1;
    else 
        noise2(i)=-1;
    end
end
i_tonal = round(i_tonal);
i_noise= round(i_noise);
tonal_bin2=i_tonal;
noise_bin2=i_noise;

t_bark = freq2bark(i_tonal,0);
tonal3 = tonal2(tonal2~=-1);

tb2 = t_bark;
n_bark = freq2bark(i_noise,0);
noise3 = noise2(noise2~=-1);
nb2 = n_bark;
end

%% 

function b = freq2bark(freq_bins,flag)
fs = 44100;
freq_arr = fs*freq_bins/512;
bark = zeros(1,length(freq_arr));
for i=1:length(bark)
    if freq_arr(i)<=1500
        bark(i)=13*atan(0.76*freq_arr(i)/1000) + 3.5*atan((freq_arr(i)/7500).^2);
    else
        bark(i)=8.7 + 14.2*log10(freq_arr(i)/1000);
        
    end
end
if flag==1
    b = round(bark);
else
    b= (bark);
end
end

%%
function ath = tq(f_bin)
f = f_bin*44100/512;
th = 3.64*((f/1000).^(-0.8))-6.5*exp(-0.6*((f/1000)-3.3).^2) + 0.010*((f/1000).^4) ;
ath = th;
end
%%
function s = SF(i,j,mask,bin,spl)
delta_z = freq2bark(i,0)-freq2bark(bin(j),0);
s1=0;
if delta_z >=-3 && delta_z <-1
    s1 = 17*delta_z - 0.4*mask(j) + 11;
elseif delta_z >=-1 && delta_z <0
    s1 = (0.4*mask(j)+6)*delta_z;
elseif delta_z >=0 && delta_z <1
    s1 = -17*delta_z;
elseif delta_z >=1 && delta_z <8
    s1 = (0.15*mask(j)-17)*delta_z - 0.15*mask(j);
end

s=s1;

end
%%
function [a,b,t_bark,n_bark,t_bin,n_bin] = indmask(tonal_dec,noise_dec,t_bin, n_bin,spl)
T_TM = zeros(256,length(t_bin));
T_NM = zeros(256,length(n_bin));
t_bark = freq2bark(t_bin,0);
bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];

for j=1:256
    for k=1:length(t_bin)
        m= round(t_bark(k));
             u = round(bw(m+1)/44100*512); % upper bound
            
             l = round(bw(m)/44100*512); % lower bound
        T_TM(j,k) = tonal_dec(k) - 0.275*freq2bark(t_bin(k),0) + SF(j,k,tonal_dec,t_bin,spl)-6.025; % tonal masker threshold
    end
end
n_bark = freq2bark(n_bin,0);

for j=1:256
    for k=1:length(n_bin)
              m= round(n_bark(k));
              
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
            
        T_NM(j,k) = noise_dec(k) - 0.175*freq2bark(n_bin(k),0) + SF(j,k,noise_dec,n_bin,spl)-2.025; % non-tonal masker threshold
    end
end

a=T_TM;
b=T_NM;
end
%%
function [tg, nfreqd] = globmaskthresh(T_TM,T_NM,X,t_bin,n_bin, t_bark, n_bark)
gmt = zeros(256,1);
f = 1:256;
Tq = tq(f);
n=0;

bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];

for j=1:256
    for k=1:length(t_bin)
        m= round(t_bark(k));
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
        T_TM(1:l-1,k)=0;
        T_TM(u+1:j,k)=0;
    end
end
n_bark = freq2bark(n_bin,0);

for j=1:256
    for k=1:length(n_bin)
              m= round(n_bark(k));
              
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
        T_NM(1:l-1,k)=0;
        T_NM(u+1:j,k)=0;
    end
end
for i=1:256
       
    gmt(i)= 10*log10(10.^(0.1.*Tq(i))+ sum(10.^(0.1.*(T_TM(i,:)))) + sum(10.^(0.1.*(T_NM(i,:))))); % global masking threshold
    if gmt(i)> X(i)
        n=n+1;
    end
end
    
tg = gmt;    
nfreqd = n;

end
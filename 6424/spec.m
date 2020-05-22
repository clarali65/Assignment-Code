clc
clear
[Y,fs]=audioread('test.wav');%load audio file
%Y is the two-channel data
%fs is sampling rate
info=audioinfo('test.wav')%return information in audio
Y1=Y(:,1);%take the second channel
figure(1)
plot(Y1)
title('original audio signal')
grid on;

figure(2)
subplot(121)
spectrogram(Y1,[rectwin(160)],80,256,16000,'yaxis');
xlabel('time(s)')
ylabel('frequency(Hz)')
title('wideband spectrogram of rectangular window')

subplot(122)
spectrogram(Y1,[rectwin(640)],320,256,16000,'yaxis');
xlabel('time(s)')
ylabel('frequency(Hz)')
title('narrowband spectrogram of rectangular window')

figure(3)
subplot(121)
spectrogram(Y1,[hann(160)],80,256,16000,'yaxis');
xlabel('time(s)')
ylabel('frequency(Hz)')
title('wideband spectrogram of Hanning window')

subplot(122)
spectrogram(Y1,[hann(640)],320,256,16000,'yaxis');
xlabel('time(s)')
ylabel('frequency(Hz)')
title('narrowband spectrogram of Hanning window')

figure(4)
subplot(121)
spectrogram(Y1,160,80,256,16000,'yaxis');
xlabel('time(s)')
ylabel('frequency(Hz)')
title('wideband spectrogram of Hamming window')

subplot(122)
spectrogram(Y1,640,320,256,16000,'yaxis');
xlabel('time(s)')
ylabel('frequency(Hz)')
title('narrowband spectrogram of Hamming window')
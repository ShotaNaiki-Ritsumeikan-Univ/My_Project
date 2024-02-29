clear all
set(groot,'DefaultLegendFontName','Times New Roman');
%% 環境設定(GPUの使用)
ExecusionEnvironment = "auto";
if(ExecusionEnvironment == "auto"&& canUseGPU)||ExecusionEnvironment == "gpu"
    gpuDevice(3)
end
%% 初期設定

load('network_230329_100epc.mat');
FS = 48000;
FFTsize = FS;
time = 1;
maxSeed = 10;
numAzimuth = 13;
numElevation = 11;
numInputChannels = 32;
numOutputChannels = 22;
intAngle = 15;
%% 正面マイクロホンの水平面の指向性のプロット
micIdx = 8;
for i = 1 : numElevation
    for j = 1 : numAzimuth

        %例外処理
            if i == numElevation
                if j ~= 1 
                    continue;
                end
            end
            if i <= 3
                if j > 4
                    continue;
                end
            end

            load(strcat("Result/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),".mat"));
            TargetChannel = Predicted(micIdx,:);
            Pow_Target = Power(TargetChannel,FS);

            if i == 5
                Result_250(j) = Pow_Target(250);
                Result_500(j) = Pow_Target(500);
                Result_1000(j) = Pow_Target(1000);
                Result_2000(j) = Pow_Target(2000);
                Result_4000(j) = Pow_Target(4000);
                Result_8000(j) = Pow_Target(8000);
                Result_16000(j) = Pow_Target(16000);
            end
    end
end

plot_250 = horzcat(Result_250,flip(Result_250));
plot_500 = horzcat(Result_500,flip(Result_500));
plot_1000 = horzcat(Result_1000,flip(Result_1000));
plot_2000 = horzcat(Result_2000,flip(Result_2000));
plot_4000 = horzcat(Result_4000,flip(Result_4000));
plot_8000 = horzcat(Result_8000,flip(Result_8000));
plot_16000 = horzcat(Result_16000,flip(Result_16000));

%% 横軸
for i = 1 : 1 : 13
    x_plot(i) = (i-1)*15;
end

Fig1 = figure(1);
plot(x_plot,plot_250)
xlabel("Angle [degs]",FontSize=13)
ylabel("Power [dB]",FontSize=13)
ylim([-70 5]);
hold on
plot(x_plot,plot_500)
hold on
plot(x_plot,plot_1000)
hold on
plot(x_plot,plot_2000)
hold on
plot(x_plot,plot_4000)
hold on
plot(x_plot,plot_8000)
hold on
plot(x_plot,plot_16000)
hold on
saveas(Fig1,"Figure/Polar_pattern.png")

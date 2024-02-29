%%環境設定(GPUの使用)
ExecusionEnvironment = "auto";
if(ExecusionEnvironment == "auto"&& canUseGPU)||ExecusionEnvironment == "gpu"
    gpuDevice(7)
end
%% 初期設定
FS = 48000;
time = 1;
maxSeed = 3462;
numAzimuth = 13;
numElevation = 11;
numInputChannels = 32;
numOutputChannels = 22;
intAngle = 15;
Win = hann(1024);
upTSP = audioread("upTSP.wav");
%% 同期加算信号の読み込みとインパルス応答の算出
% for i = 1 : numElevation
%     for j = 1 : numAzimuth
% 
%         %例外処理
%         if i == numElevation
%             if j ~= 1 
%                 continue;
%             end
%         end
%         if i <= 3
%             if j > 4
%                 continue;
%             end
%         end
% %         disp("E=");
% %         disp((i-5)*15);
% %         disp("A=");
% %         disp((j-1)*5);
% 
%         %各チャンネルごとインパルス応答を算出
%         for k = 1 : numInputChannels
%             Sync = audioread(strcat("SyncAdd/rec_E",string((i-5)*15),"A",string((j-1)*15),"_",string(k),"_5.wav"));
%             tempImpulse = conv(Sync,upTSP);
%             ImpulseLen = length(tempImpulse);
%             for l = 1 : ImpulseLen/2
%                 Impulse(l) = tempImpulse(round(ImpulseLen/2)+l);
%             end
%             save(strcat("Impulse/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"_",string(k),".mat"),"Impulse");
%         
%             %インパルス応答の窓かけ
%             WinImpulse = Impulse(1:time*FS);
%             [MAX, IDX] = max(WinImpulse);
%                  for l = IDX : IDX + 512
%                      WinImpulse(l) = WinImpulse(l) * Win(512 + l - (IDX + 1));
%                  end
%                  for l = IDX+512 : time*FS
%                      WinImpulse(l) = 0;
%                  end
%             save(strcat("WinImpulse/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"_",string(k),".mat"),"WinImpulse");
%         end
%     end
% end
% disp("Finish caliculating IR")
%% インパルス応答の窓かけ
% Win = hann(1024);
% for i = 1 : numElevation
%     for j = 1 : numAzimuth
%         %例外処理
%             if i == numElevation
%                 if j ~= 1 
%                     continue;
%                 end
%             end
%             if i <= 3
%                 if j > 4
%                     continue;
%                 end
%             end
%         disp("E=");
%         disp((i-5)*15);
%         disp("A=");
%         disp((j-1)*15);
% 
%             for k = 1 : numInputChannels
%                  load(strcat("Impulse/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"_",string(k),".mat"));
%                  WinImpulse = Impulse(1:time*FS);
%                  [MAX, IDX] = max(WinImpulse);
%                  for l = IDX : IDX + 512
%                      WinImpulse(l) = WinImpulse(l) * Win(512 + l - (IDX + 1));
%                  end
%                  for l = IDX+512 : time*FS
%                      WinImpulse(l) = 0;
%                  end
%                  save(strcat("WinImpulse/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"_",string(k),".mat"),"WinImpulse");
%                  audiowrite(strcat("WinImpulse_wav/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"_",string(k),".wav"),WinImpulse,48000);
%             end
%     end
% end

%% 入力信号の作成
for wn = 1 : maxSeed
    WhiteNoise = audioread(strcat("sound/WhiteNoise_",string(time),"_",string(wn),"seed.wav"));
    if rem(wn,10) == 0
    disp("seed =")
    disp(wn)
    end
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
    
            %各チャンネルごとに入力信号を作成
            for k = 1 : numInputChannels
                load(strcat("WinImpulse/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"_",string(k),".mat"));
                tempInput(k,:) = conv(WhiteNoise,WinImpulse);
                Len = length(tempInput);
                Input(k,:) = tempInput(1:round(Len/2));
            end
            save(strcat("Input/seed",string(wn),"/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),".mat"),"Input");
            M = max(abs(Input));
            Input = (Input .* 0.8) ./ M;
            for k = 1 : numInputChannels
                audiowrite(strcat("Input_wav/seed",string(wn),"/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"/",string(k),".wav"),Input(k,:),FS);
            end
        end
    end

%% 教師信号の作成

Elevation = [-60 -45 -30 -15 0 15 30 45 60 75 90];
Azimuth = [0 15 30 45 60 75 90 105 120 135 150 165 180];
%全チャネル制御版
matchingChannel = [19 19 18 18 18 0 0 0 0 0 0 0 0;
    5 5 4 4 4 3 3 3 2 2 2 1 1;
    13 13 12 12 12 11 11 11 10 10 10 9 9;
    17 17 17 17 17 17 17 17 17 17 17 17 17];

%単一チャネル制御版
% matchingChannel = [0 0 0 0 0 0 0 0 0 0 0 0 0;
%     5 5 0 0 0 0 0 0 0 0 0 0 0;
%     0 0 0 0 0 0 0 0 0 0 0 0 0;
%     0 0 0 0 0 0 0 0 0 0 0 0 0;];

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

            %音源方向に対応するレイヤーの算出
            if i <= 3
                matchingLayer = 1; %under
            elseif 3 < i || i <= 6
                matchingLayer = 2; %mid
            elseif 6 < i ||i <= 9
                matchingLayer = 3; %upper
            else
                matchingLayer = 4; %top
            end

            %教師信号の作成(全チャネル制御版)
            load(strcat("Input/seed",string(wn),"/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),".mat"));
            for k = 1 : numInputChannels
                if k ~= matchingChannel(matchingLayer,i)
                    Input(k,:) = 0;
                end
            end
            %教師信号の作成(1チャンネル制御版)
%             load(strcat("Input/seed",string(wn),"/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),".mat"));
%             for k = 1 : numInputChannels
%                 if k ~= matchingChannel(matchingLayer,i)
%                     Input(k,:) = 0;
%                 end
%             end
            
            save(strcat("Train/seed",string(wn),"/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),".mat"),"Input");
            M_train = max(abs(Input));
            Input = (Input .* 0.8) ./ M_train;
            for k = 1 : numInputChannels
                audiowrite(strcat("Train_wav/seed",string(wn),"/rec_E",string((i-5)*intAngle),"A",string((j-1)*intAngle),"/",string(k),".wav"),Input(k,:),FS);
            end
        end
    end
end
%% プログラムの終了
disp("End")

%% 
clear all
%% 環境設定(GPUの使用)
ExecusionEnvironment = "auto";
if(ExecusionEnvironment == "auto"&& canUseGPU)||ExecusionEnvironment == "gpu"
    gpuDevice(7)
end
% useGPUs = [5 6];
% parpool("Processes",numel(useGPUs));
% spmd
%     gpuDevice(useGPUs(spmdIndex))
% end

%% 初期設定
numInputChannels = 32;
numTrainChannels = 32;
numElevation = 11;
numAzimuth = 13;
intAngle = 15;
FS = 48000;  %サンプリング周波数
time = 1;
sigLen = FS * time;
temp_in = zeros(32,sigLen);
temp_train = zeros(22,sigLen);

%% 学習データの読み込み
tempX = fileDatastore("Input_small","ReadFcn",@load,'IncludeSubfolders',true);
%preview(tempX)
%% 教師データの読み込み
tempY = fileDatastore("Train_small","ReadFcn",@load,'IncludeSubfolders',true);
%preview(tempY)
%% データストアの変換と統合
sequenceLength = FS * time;
X = transform(tempX,@(data) struct2cell(data));
%preview(X)
Y = transform(tempY,@(data) struct2cell(data));
%preview(Y)

TrainDS = combine(X,Y);
%preview(TrainDS)
disp("Reading data finished")
%% バリデーションデータの読み込み
tempValidationX = fileDatastore("ValidationX","ReadFcn",@load,'IncludeSubfolders',true);
tempValidationY = fileDatastore("ValidationY","ReadFcn",@load,"IncludeSubfolders",true);

%バリデーションデータのデータ形式の変換
ValidationX = transform(tempValidationX,@(data) struct2cell(data));
ValidationY = transform(tempValidationY,@(data) struct2cell(data));
Validation = combine(ValidationX,ValidationY);
%% DNNの設定
numHiddenUnits = 512;   %隠れ層の総数
numResponses = 32;      %出力ノード数
numEpoch = 50;         %エポック数
Batchsize = 16;          %バッチサイズ
validationFreq = 500;   %バリデーションの間隔
LearningRate = 0.001;   %学習率
verboseFreq = 100;      %学習の途中経過の表示間隔（既定値：50）
progress = {-1};

layers  = [...
    sequenceInputLayer(numInputChannels,"Normalization","rescale-symmetric")
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    batchNormalizationLayer
    %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    %batchNormalizationLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions("adam", ...
    "MaxEpochs",numEpoch, ...
    "MiniBatchSize",Batchsize, ...
    "Plots","training-progress", ...
    "ValidationData",Validation,...
    "ValidationFrequency",validationFreq,...
    "Shuffle","every-epoch",...
    "Verbose",1, ...
    "VerboseFrequency",verboseFreq,...
    "CheckpointPath","Checkpoint", ...
    "InitialLearnRate",LearningRate);

% options = trainingOptions("adam", ...
%     "MaxEpochs",numEpoch, ...
%     "MiniBatchSize",Batchsize, ...
%     "Plots","training-progress", ...
%     "Shuffle","every-epoch",...
%     "Verbose",1, ...
%     "VerboseFrequency",verboseFreq,...
%     "CheckpointPath","Checkpoint", ...
%     "InitialLearnRate",LearningRate);

%% ネットワークの学習
 net = trainNetwork(TrainDS,layers,options);

%% モデルの保存
 save('network_test.mat','net');

 disp("end");

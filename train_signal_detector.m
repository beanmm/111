function train_signal_detector()
% 训练北斗B1I信号检测模型（全连接神经网络），支持GPU加速训练

%% 1. 配置训练参数
trainCfg = struct(...
    'fs', 40e6,   ...               % 采样率（与接收机一致）
    'IF', 98e3,    ...              % 中频（B1I默认）
    'codeLen', 2046,   ...          % 码长
    'chipRate', 2.046e6, ...        % 码速率
    'realDataPath', 'D:\update_B1l\MATLAB_SDR\IF_Data_Set\B1I.dat', ... % 去掉空格 
    'snrRange', [-15, 5], ...       % 训练信噪比范围（含弱信号）
    'prnList', 1:37,   ...          % 北斗B1I PRN范围
    'numRealSamples', 5000, ...     % 从真实数据中提取的样本数
    'numSyntheticSamples', 8000, ... % 合成样本数（用于数据增强）
    'valRatio', 0.2,    ...         % 验证集比例
    'batchSize', 64, ...            % 批大小（适配GPU显存）
    'epochs', 30,  ...              % 训练轮次
    'lr', 1e-3,        ...          % 初始学习率
    'modelSavePath', './ML_Models/signal_detector.onnx'...
);

% 创建模型保存目录
modelDir = fileparts(trainCfg.modelSavePath);
if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end

%% 2. 从真实数据集中提取训练样本
fprintf('Loading real dataset from: %s\n', trainCfg.realDataPath);
[realTrainData, realValData] = extract_samples_from_real_data(trainCfg);

%% 3. 生成合成样本用于数据增强
fprintf('Generating synthetic samples for data augmentation...\n');
[synthTrainData, synthValData] = generate_synthetic_dataset(trainCfg);

%% 4. 合并真实数据和合成数据，并进行标准化
% 详细检查数据分布和维度
fprintf('=== 数据分布和维度检查 ===\n');

% 检查真实数据
if isempty(realTrainData.features)
    fprintf('真实训练数据: 空\n');
else
    fprintf('真实训练数据: %d样本, %d特征, 信号比例: %.1f%%\n', ...
        size(realTrainData.features, 1), size(realTrainData.features, 2), ...
        sum(realTrainData.labels == 'signal')/length(realTrainData.labels)*100);
end

if isempty(realValData.features)
    fprintf('真实验证数据: 空\n');
else
    fprintf('真实验证数据: %d样本, %d特征, 信号比例: %.1f%%\n', ...
        size(realValData.features, 1), size(realValData.features, 2), ...
        sum(realValData.labels == 'signal')/length(realValData.labels)*100);
end

% 检查合成数据
fprintf('合成训练数据: %d样本, %d特征, 信号比例: %.1f%%\n', ...
    size(synthTrainData.features, 1), size(synthTrainData.features, 2), ...
    sum(synthTrainData.labels == 'signal')/length(synthTrainData.labels)*100);

fprintf('合成验证数据: %d样本, %d特征, 信号比例: %.1f%%\n', ...
    size(synthValData.features, 1), size(synthValData.features, 2), ...
    sum(synthValData.labels == 'signal')/length(synthValData.labels)*100);



% 智能数据合并 - 创建更现实的训练/验证分割
realTrainRatio = sum(realTrainData.labels == 'signal')/length(realTrainData.labels);
realValRatio = sum(realValData.labels == 'signal')/length(realValData.labels);

fprintf('真实训练数据信号比例: %.1f%%\n', realTrainRatio*100);
fprintf('真实验证数据信号比例: %.1f%%\n', realValRatio*100);

% 创建更现实的验证集，避免与训练集过于相似
if isempty(realTrainData.features) || size(realTrainData.features, 1) == 0
    fprintf('真实数据为空，只使用合成数据（创建独立验证集）\n');
    
    % 使用完全不同的合成数据子集作为验证集
    totalSynthSamples = size(synthTrainData.features, 1);
    
    % 随机分割合成数据为训练集和验证集（70%训练，30%验证）
    allIndices = randperm(totalSynthSamples);
    trainIndices = allIndices(1:floor(0.7 * totalSynthSamples));
    valIndices = allIndices(floor(0.7 * totalSynthSamples) + 1:end);
    
    trainData.features = synthTrainData.features(trainIndices, :);
    trainData.labels = synthTrainData.labels(trainIndices);
    valData.features = synthTrainData.features(valIndices, :);
    valData.labels = synthValData.labels(valIndices);
    
    fprintf('使用独立数据分割: 训练%d样本，验证%d样本\n', ...
        length(trainData.labels), length(valData.labels));
    
elseif realTrainRatio > 0.9  % 如果真实数据信号比例过高（>90%）
    fprintf('真实数据信号比例过高(%.1f%%)，采用混合策略\n', realTrainRatio*100);
    
    % 混合真实数据和合成数据，但创建独立的验证集
    targetRealSamples = min(800, size(realTrainData.features, 1));  % 减少真实样本
    targetSynthSamples = 1200;  % 增加合成样本多样性
    
    % 训练集：混合真实和合成数据
    realIdx = randperm(size(realTrainData.features, 1), targetRealSamples);
    realFeatures = realTrainData.features(realIdx, :);
    realLabels = realTrainData.labels(realIdx);
    
    synthTrainIdx = randperm(size(synthTrainData.features, 1), targetSynthSamples);
    trainData.features = [realFeatures; synthTrainData.features(synthTrainIdx, :)];
    trainData.labels = [realLabels; synthTrainData.labels(synthTrainIdx)];
    
    % 验证集：使用完全不同的数据子集
    if ~isempty(realValData.features) && size(realValData.features, 1) > 0
        % 使用真实验证数据，但限制数量
        valRealCount = min(300, size(realValData.features, 1));
        valRealIdx = randperm(size(realValData.features, 1), valRealCount);
        valRealFeatures = realValData.features(valRealIdx, :);
        valRealLabels = realValData.labels(valRealIdx);
        
        % 从合成验证数据中选择不同子集
        valSynthCount = 500;  % 增加验证集大小
        valSynthIdx = randperm(size(synthValData.features, 1), valSynthCount);
        valData.features = [valRealFeatures; synthValData.features(valSynthIdx, :)];
        valData.labels = [valRealLabels; synthValData.labels(valSynthIdx)];
    else
        % 只使用合成数据，但确保与训练集不同
        valSynthCount = 800;
        % 选择与训练集不同的合成数据子集
        allSynthIdx = randperm(size(synthValData.features, 1));
        valSynthIdx = allSynthIdx(1:min(valSynthCount, length(allSynthIdx)));
        
        valData.features = synthValData.features(valSynthIdx, :);
        valData.labels = synthValData.labels(valSynthIdx);
        
        fprintf('警告: 无真实验证数据，使用独立合成验证集: %d样本\n', length(valData.labels));
    end
else
    fprintf('真实数据信号比例合理(%.1f%%)，创建独立训练/验证分割\n', realTrainRatio*100);
    
    % 创建更严格的训练/验证分割
    % 训练集：使用部分真实和合成数据
    trainRealCount = min(1000, size(realTrainData.features, 1));
    trainSynthCount = 1500;
    
    realTrainIdx = randperm(size(realTrainData.features, 1), trainRealCount);
    synthTrainIdx = randperm(size(synthTrainData.features, 1), trainSynthCount);
    
    trainData.features = [realTrainData.features(realTrainIdx, :); synthTrainData.features(synthTrainIdx, :)];
    trainData.labels = [realTrainData.labels(realTrainIdx); synthTrainData.labels(synthTrainIdx)];
    
    % 验证集：使用不同的数据子集
    if ~isempty(realValData.features) && size(realValData.features, 1) > 0
        valRealCount = min(500, size(realValData.features, 1));
        valRealIdx = randperm(size(realValData.features, 1), valRealCount);
        
        valSynthCount = 800;
        valSynthIdx = randperm(size(synthValData.features, 1), valSynthCount);
        
        valData.features = [realValData.features(valRealIdx, :); synthValData.features(valSynthIdx, :)];
        valData.labels = [realValData.labels(valRealIdx); synthValData.labels(valSynthIdx)];
    else
        % 使用与训练集不同的合成数据子集
        valSynthCount = 1000;
        totalSynthSize = size(synthValData.features, 1);
        startIdx = trainSynthCount + 1;  % 从训练集之后开始
        if startIdx + valSynthCount <= totalSynthSize
            valSynthIdx = startIdx:startIdx + valSynthCount - 1;
        else
            valSynthIdx = randperm(totalSynthSize, min(valSynthCount, totalSynthSize));
        end
        
        valData.features = synthValData.features(valSynthIdx, :);
        valData.labels = synthValData.labels(valSynthIdx);
    end
end

% 显示合并后的数据分布
fprintf('\n=== 合并后数据分布 ===\n');
fprintf('训练集: %d样本, 信号: %d (%.1f%%), 噪声: %d (%.1f%%)\n', ...
    length(trainData.labels), ...
    sum(trainData.labels == 'signal'), ...
    sum(trainData.labels == 'signal')/length(trainData.labels)*100, ...
    sum(trainData.labels == 'noise'), ...
    sum(trainData.labels == 'noise')/length(trainData.labels)*100);
fprintf('验证集: %d样本, 信号: %d (%.1f%%), 噪声: %d (%.1f%%)\n', ...
    length(valData.labels), ...
    sum(valData.labels == 'signal'), ...
    sum(valData.labels == 'signal')/length(valData.labels)*100, ...
    sum(valData.labels == 'noise'), ...
    sum(valData.labels == 'noise')/length(valData.labels)*100);

%% 5. 定义更复杂的全连接神经网络架构（增加正则化）
inputDim = size(trainData.features, 2);
layers = [
    % 输入层
    featureInputLayer(inputDim, 'Name', 'input')
    
    % 第一个全连接层
    fullyConnectedLayer(128, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.4, 'Name', 'dropout1')
    
    % 第二个全连接层
    fullyConnectedLayer(64, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.4, 'Name', 'dropout2')
    
    % 第三个全连接层
    fullyConnectedLayer(32, 'Name', 'fc3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.3, 'Name', 'dropout3')
    
    % 第四个全连接层
    fullyConnectedLayer(16, 'Name', 'fc4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.3, 'Name', 'dropout4')
    
    % 输出层
    fullyConnectedLayer(2, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% 6. 调整训练参数 - 使用更保守的设置避免NaN
% 数据标准化
fprintf('进行数据标准化...\n');
featureMean = mean(trainData.features, 1);
featureStd = std(trainData.features, 0, 1) + 1e-8; % 添加小常数避免除零
trainData.features = (trainData.features - featureMean) ./ featureStd;
valData.features = (valData.features - featureMean) ./ featureStd;

% 数据增强 - 添加适度噪声增加难度
fprintf('添加数据增强...\n');
noiseLevel = 0.08;  % 增加噪声水平
augmentedFeatures = trainData.features + noiseLevel * randn(size(trainData.features));
trainData.features = [trainData.features; augmentedFeatures];
trainData.labels = [trainData.labels; trainData.labels];

% 添加额外的数据增强 - 轻微的特征扰动
fprintf('添加特征扰动增强...\n');
perturbLevel = 0.03;  % 特征扰动水平
perturbedFeatures = trainData.features .* (1 + perturbLevel * randn(size(trainData.features)));
trainData.features = [trainData.features; perturbedFeatures];
trainData.labels = [trainData.labels; trainData.labels];

% 保存标准化参数和数据集用于分析
save('./ML_Models/feature_stats.mat', 'featureMean', 'featureStd', 'trainData', 'valData');

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...                     % 增加训练轮数
    'InitialLearnRate', 1e-4, ...            % 提高学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...           % 增加验证频率
    'MiniBatchSize', 32, ...                 % 减小批次大小增加随机性
    'ValidationData', {valData.features, valData.labels}, ...
    'ValidationFrequency', 20, ...           % 增加验证频率
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose', true, ...
    'VerboseFrequency', 20, ...
    'ValidationPatience', 5);                % 保持早停机制
%% 7. 模型训练
fprintf('Starting model training...\n');
net = trainNetwork(trainData.features, trainData.labels, layers, options);

%% 8. 模型评估与导出
% 验证性能评估
valPred = classify(net, valData.features);
valAcc = mean(valPred == valData.labels);
valConfMat = confusionmat(valData.labels, valPred);

fprintf('\n=== 模型性能评估 ===\n');
fprintf('验证准确率: %.2f%%\n', valAcc*100);
fprintf('混淆矩阵:\n');
disp(valConfMat);

% === 北斗B1I信号检测模型完整评估指标推导 ===
fprintf('\n=== 北斗B1I信号检测模型评估指标推导 ===\n');
fprintf('基于混淆矩阵的4个核心变量:\n');
fprintf('TP (True Positive): 真实信号被正确检测为信号\n');
fprintf('TN (True Negative): 真实噪声被正确检测为噪声\n'); 
fprintf('FP (False Positive): 真实噪声被错误检测为信号（误报）\n');
fprintf('FN (False Negative): 真实信号被错误检测为噪声（漏检）\n\n');

if size(valConfMat, 1) == 2 && size(valConfMat, 2) == 2
    % 提取混淆矩阵的4个核心变量
    % 假设: 第1行=噪声, 第2行=信号; 第1列=噪声, 第2列=信号
    TN = valConfMat(1,1);  % 真实噪声预测为噪声
    FP = valConfMat(1,2);  % 真实噪声预测为信号  
    FN = valConfMat(2,1);  % 真实信号预测为噪声
    TP = valConfMat(2,2);  % 真实信号预测为信号
    
    fprintf('混淆矩阵核心变量:\n');
    fprintf('TP = %d, TN = %d, FP = %d, FN = %d\n', TP, TN, FP, FN);
    fprintf('总样本数 = %d\n\n', TP + TN + FP + FN);
    
    % 1. 准确率 (Accuracy) - 已计算
    fprintf('1. 准确率 (Accuracy):\n');
    fprintf('   公式: (TP + TN) / (TP + TN + FP + FN)\n');
    fprintf('   计算: (%d + %d) / (%d + %d + %d + %d) = %.4f\n', ...
            TP, TN, TP, TN, FP, FN, valAcc);
    fprintf('   含义: 所有预测中正确的比例\n\n');
    
    % 2. 检测率/召回率 (Detection Rate / Recall)
    if (TP + FN) > 0
        detection_rate = TP / (TP + FN);
    else
        detection_rate = 0;
    end
    fprintf('2. 检测率/召回率 (Detection Rate / Recall):\n');
    fprintf('   公式: TP / (TP + FN)\n');
    fprintf('   计算: %d / (%d + %d) = %.4f\n', TP, TP, FN, detection_rate);
    fprintf('   含义: 所有真实信号中被正确检测出的比例\n\n');
    
    % 3. 误报率 (False Alarm Rate)
    if (FP + TN) > 0
        false_alarm_rate = FP / (FP + TN);
    else
        false_alarm_rate = 0;
    end
    fprintf('3. 误报率 (False Alarm Rate):\n');
    fprintf('   公式: FP / (FP + TN)\n');
    fprintf('   计算: %d / (%d + %d) = %.4f\n', FP, FP, TN, false_alarm_rate);
    fprintf('   含义: 所有真实噪声中被错误检测为信号的比例\n\n');
    
    % 4. 精确率 (Precision)
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    fprintf('4. 精确率 (Precision):\n');
    fprintf('   公式: TP / (TP + FP)\n');
    fprintf('   计算: %d / (%d + %d) = %.4f\n', TP, TP, FP, precision);
    fprintf('   含义: 所有预测为信号中真正是信号的比例\n\n');
    
    % 5. F1分数 (F1-Score)
    if (precision + detection_rate) > 0
        f1_score = 2 * (precision * detection_rate) / (precision + detection_rate);
    else
        f1_score = 0;
    end
    fprintf('5. F1分数 (F1-Score):\n');
    fprintf('   公式: 2 * (Precision * Recall) / (Precision + Recall)\n');
    fprintf('   计算: 2 * (%.4f * %.4f) / (%.4f + %.4f) = %.4f\n', ...
            precision, detection_rate, precision, detection_rate, f1_score);
    fprintf('   含义: 精确率和召回率的调和平均数\n\n');
    
    % 汇总结果
    fprintf('=== 北斗B1I信号检测评估指标汇总 ===\n');
    fprintf('准确率 (Accuracy):     %.2f%%\n', valAcc * 100);
    fprintf('检测率 (Detection Rate): %.2f%%\n', detection_rate * 100);
    fprintf('误报率 (False Alarm Rate): %.2f%%\n', false_alarm_rate * 100);
    fprintf('精确率 (Precision):    %.2f%%\n', precision * 100);
    fprintf('F1分数 (F1-Score):     %.4f\n', f1_score);
    
    % 北斗B1I信号检测特定指标解释
    fprintf('\n=== 北斗B1I信号检测特定指标解释 ===\n');
    fprintf('检测率 (%.1f%%): 每100个真实信号中，能正确检测出%.0f个\n', ...
            detection_rate * 100, detection_rate * 100);
    fprintf('误报率 (%.1f%%): 每100个真实噪声中，会有%.0f个被错误检测为信号\n', ...
            false_alarm_rate * 100, false_alarm_rate * 100);
    fprintf('精确率 (%.1f%%): 每100个预测为信号的结果中，有%.0f个是真正的信号\n', ...
            precision * 100, precision * 100);
    
else
    fprintf('无法计算当前混淆矩阵的评估指标\n');
    precision = 0;
    detection_rate = 0;
    false_alarm_rate = 0;
    f1_score = 0;
end

% 进行诚实的特征可分性分析
fprintf('\n=== 诚实评估 - 特征可分性分析 ===\n');
all_features = [trainData.features; valData.features];
all_labels = [trainData.labels; valData.labels];
signal_feat = all_features(all_labels == 'signal', :);
noise_feat = all_features(all_labels == 'noise', :);
signal_mean = mean(signal_feat);
noise_mean = mean(noise_feat);
diff = abs(signal_mean - noise_mean);

fprintf('信号样本数: %d, 噪声样本数: %d\n', size(signal_feat, 1), size(noise_feat, 1));
fprintf('特征维度: %d\n', size(signal_feat, 2));
fprintf('平均特征差异: %.4f\n', mean(diff));
fprintf('最大特征差异: %.4f\n', max(diff));
fprintf('特征差异>1的比例: %.1f%%\n', 100*sum(diff > 1)/length(diff));

% 简单线性分类器测试
fprintf('\n=== 线性可分性测试 ===\n');
w = signal_mean - noise_mean;  % 权重向量
b = -0.5 * (signal_mean + noise_mean) * w';  % 偏置
linear_pred = all_features * w' + b > 0;
true_binary = all_labels == 'signal';
linear_accuracy = mean(linear_pred == true_binary);
fprintf('简单线性分类器准确率: %.2f%%\n', linear_accuracy*100);

% 提供现实性能预期
fprintf('\n=== 现实性能预期 ===\n');
if valAcc >= 0.99 && linear_accuracy >= 0.99
    fprintf('⚠️  警告: 检测到极高的准确率！\n');
    fprintf('   - 合成数据特征分离度过高（%.2f）\n', mean(diff));
    fprintf('   - 简单线性分类器也能达到%.1f%%准确率\n', linear_accuracy*100);
    fprintf('   - 这些结果**不代表**模型在真实数据上的表现\n');
    fprintf('   - 真实数据可能包含更多噪声、重叠和复杂性\n');
    fprintf('   - 建议: 需要真实数据来验证模型性能\n');
    fprintf('   - 现实预期: 真实数据准确率70-90%（而非100%）\n');
elseif valAcc >= 0.95
    fprintf('⚠️  注意: 准确率较高，但需要谨慎解读\n');
    fprintf('   - 合成数据可能过于理想化\n');
    fprintf('   - 建议在真实数据上进一步验证\n');
else
    fprintf('✓ 性能指标在合理范围内\n');
    fprintf('   - 模型显示出了一定的学习能力\n');
    fprintf('   - 但仍需在真实数据上验证\n');
end

% 修正变量名错误
try
    exportONNXNetwork(net, trainCfg.modelSavePath);
    fprintf('ONNX Model saved to: %s\n', trainCfg.modelSavePath);
catch ME
    fprintf('ONNX导出失败: %s\n', ME.message);
    % 保存为MAT文件
    matSavePath = strrep(trainCfg.modelSavePath, '.onnx', '.mat');
    save(matSavePath, 'net', '-v7.3');
    fprintf('模型已保存为MAT文件: %s\n', matSavePath);
end

% 保存训练报告（包含诚实评估信息和完整评估指标）
save_training_report(trainCfg, valAcc, precision, detection_rate, f1_score, linear_accuracy, mean(diff));

%% 9. 动态可视化测试阶段（新增功能）
fprintf('\n=== 动态可视化测试阶段 ===\n');
fprintf('正在启动动态测试与可视化评估...\n');

% 调用动态测试函数（如果存在）
if exist('dynamic_evaluation_with_visualization', 'file')
    try
        dynamic_evaluation_with_visualization(net, valData, featureMean, featureStd);
    catch ME
        fprintf('动态测试失败: %s\n', ME.message);
        fprintf('使用简化版测试...\n');
        simple_dynamic_evaluation(net, valData);
    end
else
    % 使用内置简化动态测试
    simple_dynamic_evaluation(net, valData);
end

fprintf('Training completed successfully!\n');

end

%% 内置简化动态测试函数
function simple_dynamic_evaluation(net, testData)
% 简化版动态测试函数

fprintf('\n>>> 启动简化动态测试 <<<\n');

% 创建图形窗口
fig = figure('Name', '北斗B1I信号检测 - 简化动态评估', 'Position', [200, 200, 1000, 600]);

% 批量预测
fprintf('正在进行批量预测...\n');
[predictions, probabilities] = classify(net, testData.features);
signal_probabilities = probabilities(2, :);

% 计算评估指标
true_binary = strcmp(testData.labels, 'signal');
pred_binary = strcmp(predictions, 'signal');

% 混淆矩阵4个核心变量
TP = sum(true_binary & pred_binary);  % 真实信号预测为信号
TN = sum(~true_binary & ~pred_binary);  % 真实噪声预测为噪声
FP = sum(~true_binary & pred_binary);  % 真实噪声预测为信号（误报）
FN = sum(true_binary & ~pred_binary);  % 真实信号预测为噪声（漏检）

fprintf('\n=== 动态测试评估指标推导 ===\n');
fprintf('混淆矩阵核心变量:\n');
fprintf('TP = %d, TN = %d, FP = %d, FN = %d\n', TP, TN, FP, FN);
fprintf('总测试样本数 = %d\n\n', TP + TN + FP + FN);

% 北斗B1I信号检测完整评估指标计算
accuracy = (TP + TN) / (TP + TN + FP + FN);
detection_rate = TP / (TP + FN);  % 检测率/召回率
false_alarm_rate = FP / (FP + TN);  % 误报率
precision = TP / (TP + FP);  % 精确率

if (precision + detection_rate) > 0
    f1_score = 2 * (precision * detection_rate) / (precision + detection_rate);
else
    f1_score = 0;
end

fprintf('动态测试评估指标:\n');
fprintf('1. 准确率 (Accuracy): %.2f%%\n', accuracy * 100);
fprintf('   计算: (%d + %d) / (%d + %d + %d + %d) = %.4f\n', TP, TN, TP, TN, FP, FN, accuracy);
fprintf('2. 检测率 (Detection Rate): %.2f%%\n', detection_rate * 100);
fprintf('   计算: %d / (%d + %d) = %.4f\n', TP, TP, FN, detection_rate);
fprintf('3. 误报率 (False Alarm Rate): %.2f%%\n', false_alarm_rate * 100);
fprintf('   计算: %d / (%d + %d) = %.4f\n', FP, FP, TN, false_alarm_rate);
fprintf('4. 精确率 (Precision): %.2f%%\n', precision * 100);
fprintf('   计算: %d / (%d + %d) = %.4f\n', TP, TP, FP, precision);
fprintf('5. F1分数 (F1-Score): %.4f\n', f1_score);
fprintf('   计算: 2 * (%.4f * %.4f) / (%.4f + %.4f) = %.4f\n', ...
        precision, detection_rate, precision, detection_rate, f1_score);

%% 创建可视化图表
clf(fig);

% 子图1: 混淆矩阵
subplot(2, 3, 1);
conf_matrix = [TN, FP; FN, TP];
imagesc(conf_matrix);
colorbar;
colormap(bone); % 使用内置的bone colormap替代Blues
title('混淆矩阵');
xlabel('预测类别');
ylabel('真实类别');
set(gca, 'XTick', 1:2, 'XTickLabel', {'噪声', '信号'});
set(gca, 'YTick', 1:2, 'YTickLabel', {'噪声', '信号'});

% 添加数值标注
for i = 1:2
    for j = 1:2
        text(j, i, num2str(conf_matrix(i, j)), 'HorizontalAlignment', 'center', 'FontSize', 12);
    end
end

% 子图2: 预测概率分布
subplot(2, 3, 2);
[counts, bins] = histcounts(signal_probabilities, 20);
bar(bins(1:end-1), counts, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'none');
xlabel('信号预测概率');
ylabel('样本数量');
title('预测概率分布');
grid on;

% 子图3: 性能指标柱状图
subplot(2, 3, 3);
metrics = [accuracy, detection_rate, precision, f1_score] * 100;
metric_names = {'准确率', '检测率', '精确率', 'F1分数'};
colors = [0.2 0.8 0.2; 0.8 0.2 0.2; 0.2 0.2 0.8; 0.8 0.8 0.2];
bar(metrics, 'FaceColor', 'flat', 'CData', colors);
set(gca, 'XTickLabel', metric_names);
ylabel('性能指标 (%)');
title('核心性能指标');
ylim([0 110]);
grid on;

% 添加数值标注
for i = 1:length(metrics)
    text(i, metrics(i) + 2, sprintf('%.1f%%', metrics(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

% 子图4: 样本类型分布饼图
subplot(2, 3, 4);
labels = {'信号', '噪声'};
sizes = [TP + FN, TN + FP];
pie(sizes, labels);
title(sprintf('样本分布\n总样本: %d', length(testData.labels)));

% 子图5: 性能指标表格
subplot(2, 3, 5);
axis off;

% 创建性能表格
performance_text = sprintf([...
    '性能评估结果\n\n' ...
    '准确率 (Accuracy): %.2f%%\n' ...
    '检测率 (DR): %.2f%%\n' ...
    '误报率 (FAR): %.2f%%\n' ...
    '精确率 (Precision): %.2f%%\n' ...
    'F1分数 (F1-Score): %.3f\n\n' ...
    '混淆矩阵:\n' ...
    'TN: %d, FP: %d\n' ...
    'FN: %d, TP: %d\n\n' ...
    '样本统计:\n' ...
    '总样本: %d\n' ...
    '信号: %d (%.1f%%)\n' ...
    '噪声: %d (%.1f%%)'], ...
    accuracy * 100, detection_rate * 100, false_alarm_rate * 100, ...
    precision * 100, f1_score, TN, FP, FN, TP, ...
    length(testData.labels), TP + FN, mean(true_binary) * 100, ...
    TN + FP, (1 - mean(true_binary)) * 100);

text(0.1, 0.9, performance_text, 'FontSize', 10, 'VerticalAlignment', 'top');

% 子图6: 性能等级评估
subplot(2, 3, 6);
axis off;

% 评估性能等级
if detection_rate >= 0.9 && false_alarm_rate <= 0.05
    grade = '优秀';
    color = [0 0.8 0];
    message = '检测率和误报率均达标！';
elself detection_rate >= 0.8 && false_alarm_rate <= 0.1
    grade = '良好';
    color = [1 0.8 0];
    message = '性能良好，有改进空间。';
else
    grade = '需改进';
    color = [0.8 0 0];
    message = '需要优化模型参数。';
end

% 绘制等级评估
text(0.5, 0.7, grade, 'HorizontalAlignment', 'center', 'FontSize', 24, ...
    'FontWeight', 'bold', 'Color', color);
text(0.5, 0.4, message, 'HorizontalAlignment', 'center', 'FontSize', 12);
text(0.5, 0.2, sprintf('综合评分: %.1f/100', (f1_score * 100)), ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

% 控制台输出结果
fprintf('\n=== 简化动态测试结果 ===\n');
fprintf('准确率: %.2f%%\n', accuracy * 100);
fprintf('检测率: %.2f%%\n', detection_rate * 100);
fprintf('误报率: %.2f%%\n', false_alarm_rate * 100);
fprintf('精确率: %.2f%%\n', precision * 100);
fprintf('F1分数: %.3f\n', f1_score);

% 北斗B1I信号检测性能分析
fprintf('\n=== 北斗B1I信号检测性能分析 ===\n');
fprintf('检测率 (%.1f%%): 每100个真实信号中，能正确检测出%.0f个\n', ...
        detection_rate * 100, detection_rate * 100);
fprintf('误报率 (%.1f%%): 每100个真实噪声中，会有%.0f个被错误检测为信号\n', ...
        false_alarm_rate * 100, false_alarm_rate * 100);
fprintf('精确率 (%.1f%%): 每100个预测为信号的结果中，有%.0f个是真正的信号\n', ...
        precision * 100, precision * 100);
fprintf('\n混淆矩阵: [TN:%d FP:%d; FN:%d TP:%d]\n', TN, FP, FN, TP);
fprintf('性能等级: %s\n', grade);

% 添加诚实评估警告
if accuracy >= 0.99
    fprintf('\n⚠️  诚实评估警告:\n');
    fprintf('   当前测试在合成数据上进行，可能过于乐观\n');
    fprintf('   真实数据通常包含更多噪声和重叠\n');
    fprintf('   建议在真实数据上进一步验证模型性能\n');
    fprintf('   现实预期准确率: 70-90%%（而非100%%）\n');
end

end
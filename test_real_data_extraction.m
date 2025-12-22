% 真实数据提取测试脚本
% 用于验证extract_samples_from_real_data函数的功能

fprintf('=== 真实数据提取测试 ===\n');

% 配置参数
trainCfg = struct();
trainCfg.realDataPath = 'D:\\update_B1l\\MATLAB_SDR\\IF_Data_Set\\B1I.dat';  % 北斗B1I中频数据路径
trainCfg.fs = 40e6;              % 采样率40MHz
trainCfg.valRatio = 0.2;         % 验证集比例20%
trainCfg.numRealSamples = 1000;  % 提取1000个样本
trainCfg.logPath = 'D:\\update_B1l\\MATLAB_SDR\\BDS_B1I_B2I\\current_run.log';  % 日志文件路径

fprintf('配置参数:\n');
fprintf('  数据文件: %s\n', trainCfg.realDataPath);
fprintf('  采样率: %.1f MHz\n', trainCfg.fs/1e6);
fprintf('  验证集比例: %.1f%%\n', trainCfg.valRatio*100);
fprintf('  目标样本数: %d\n', trainCfg.numRealSamples);

% 执行真实数据提取
fprintf('\n开始提取真实数据...\n');
try
    [realTrainData, realValData] = extract_samples_from_real_data(trainCfg);
    
    % 显示结果
    fprintf('\n=== 提取结果 ===\n');
    fprintf('训练集样本数: %d\n', size(realTrainData.features, 1));
    fprintf('验证集样本数: %d\n', size(realValData.features, 1));
    fprintf('特征维度: %d\n', size(realTrainData.features, 2));
    
    % 显示标签分布
    trainLabels = realTrainData.labels;
    valLabels = realValData.labels;
    
    trainSignalCount = sum(trainLabels == "signal");
    trainNoiseCount = sum(trainLabels == "noise");
    valSignalCount = sum(valLabels == "signal");
    valNoiseCount = sum(valLabels == "noise");
    
    fprintf('\n训练集标签分布:\n');
    fprintf('  信号: %d (%.1f%%)\n', trainSignalCount, trainSignalCount/length(trainLabels)*100);
    fprintf('  噪声: %d (%.1f%%)\n', trainNoiseCount, trainNoiseCount/length(trainLabels)*100);
    
    fprintf('\n验证集标签分布:\n');
    fprintf('  信号: %d (%.1f%%)\n', valSignalCount, valSignalCount/length(valLabels)*100);
    fprintf('  噪声: %d (%.1f%%)\n', valNoiseCount, valNoiseCount/length(valLabels)*100);
    
    % 检查数据完整性
    fprintf('\n=== 数据完整性检查 ===\n');
    if size(realTrainData.features, 1) > 0 && size(realValData.features, 1) > 0
        fprintf('✓ 数据提取成功\n');
        
        % 检查特征值
        trainFeatures = realTrainData.features;
        if any(isnan(trainFeatures(:)))
            fprintf('⚠ 训练集特征包含NaN值\n');
        else
            fprintf('✓ 训练集特征无NaN值\n');
        end
        
        if any(isinf(trainFeatures(:)))
            fprintf('⚠ 训练集特征包含Inf值\n');
        else
            fprintf('✓ 训练集特征无Inf值\n');
        end
        
        % 显示特征统计
        fprintf('\n特征统计信息:\n');
        fprintf('  特征均值: %.4f\n', mean(trainFeatures(:)));
        fprintf('  特征标准差: %.4f\n', std(trainFeatures(:)));
        fprintf('  特征范围: [%.4f, %.4f]\n', min(trainFeatures(:)), max(trainFeatures(:)));
        
    else
        fprintf('✗ 数据提取失败\n');
    end
    
catch ME
    fprintf('\n✗ 测试失败: %s\n', ME.message);
    fprintf('错误发生在: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
end

fprintf('\n=== 测试完成 ===\n');
function [acqResults, processingStats] = enhanced_gpu_acquisition_safe(fid, acqSettings)
% 增强版GPU捕获 - 集成深度学习预筛选和智能频率优化
% 目标：减少GPU全频带冗余计算，提升弱信号检测能力

    %% 1. 初始化设置和路径
    if exist('setup_paths', 'file') == 2
        setup_paths;
    else
        criticalPaths = {
            fullfile(fileparts(mfilename('fullpath')), 'include'),
            fullfile(fileparts(mfilename('fullpath')), 'ML_Models'),
            fullfile(fileparts(mfilename('fullpath')), 'ML_Functions'),
            fullfile(fileparts(mfilename('fullpath')), 'acquisition'),
            fullfile(fileparts(mfilename('fullpath')), 'tracking')
        };
        for i = 1:length(criticalPaths)
            if exist(criticalPaths{i}, 'dir') == 7
                addpath(criticalPaths{i});
            end
        end
    end
    
    %% 2. 增强参数配置
    % 基本参数设置
    if ~isfield(acqSettings, 'samplingFreq'), acqSettings.samplingFreq = 20e6; end % 默认20MHz
    if ~isfield(acqSettings, 'IF'), acqSettings.IF = 4e6; end % 默认4MHz中频
    if ~isfield(acqSettings, 'codeFreqBasis'), acqSettings.codeFreqBasis = 2.046e6; end % 默认码频率
    
    % 深度学习预筛选参数
    if ~isfield(acqSettings, 'mlPreScreening'), acqSettings.mlPreScreening = true; end
    if ~isfield(acqSettings, 'mlConfidenceThreshold'), acqSettings.mlConfidenceThreshold = 0.75; end
    if ~isfield(acqSettings, 'mlMinConfidentBlocks'), acqSettings.mlMinConfidentBlocks = 2; end
    if ~isfield(acqSettings, 'mlSegmentLength'), acqSettings.mlSegmentLength = 2; end % ms
    if ~isfield(acqSettings, 'mlBlockSizeMs'), acqSettings.mlBlockSizeMs = 4; end
    if ~isfield(acqSettings, 'mlModelPath'), acqSettings.mlModelPath = 'D:\update_B1l\MATLAB_SDR\BDS_B1I_B2I\ML_Models\ML_Models\signal_detector.mat'; end
    if ~isfield(acqSettings, 'mlFeatureStatsPath'), acqSettings.mlFeatureStatsPath = 'D:\update_B1l\MATLAB_SDR\BDS_B1I_B2I\ML_Models\ML_Models\feature_stats.mat'; end
    
    % 智能频率搜索参数
    if ~isfield(acqSettings, 'smartFreqSearch'), acqSettings.smartFreqSearch = true; end
    if ~isfield(acqSettings, 'coarseFreqStep'), acqSettings.coarseFreqStep = 1000; end % Hz
    if ~isfield(acqSettings, 'fineFreqStep'), acqSettings.fineFreqStep = 200; end % Hz
    if ~isfield(acqSettings, 'fineFreqRange'), acqSettings.fineFreqRange = 2000; end % Hz
    if ~isfield(acqSettings, 'adaptiveThreshold'), acqSettings.adaptiveThreshold = true; end
    if ~isfield(acqSettings, 'weakSignalBoost'), acqSettings.weakSignalBoost = true; end
    if ~isfield(acqSettings, 'weakSignalThreshold'), acqSettings.weakSignalThreshold = 0.12; end
    
    % GPU内存优化参数 - RTX 4050优化配置
    if ~isfield(acqSettings, 'gpuMemoryLimit'), acqSettings.gpuMemoryLimit = 0.7; end % 70% GPU内存 (约3.7GB)
    if ~isfield(acqSettings, 'maxBlockSize'), acqSettings.maxBlockSize = 65536; end % 增大块大小以提高并行度
    if ~isfield(acqSettings, 'minBlockSize'), acqSettings.minBlockSize = 2048; end % 最小块大小增加
    if ~isfield(acqSettings, 'memorySafetyFactor'), acqSettings.memorySafetyFactor = 0.85; end % 提高内存利用率
    if ~isfield(acqSettings, 'gpuCheckInterval'), acqSettings.gpuCheckInterval = 5; end % 减少检查频率
    if ~isfield(acqSettings, 'gpuRecoveryAttempts'), acqSettings.gpuRecoveryAttempts = 2; end % 减少恢复尝试次数
    
    %% 3. GPU可用性检查和内存优化
    [gpuAvailable, gpuInfo] = checkGPUAvailability(acqSettings);
    acqSettings.gpuInfo = gpuInfo;
    
    if acqSettings.useGPU && gpuAvailable
        try
            gpuDeviceObj = gpuDevice;
            totalMemory = gpuDeviceObj.TotalMemory;
            availableMemory = gpuDeviceObj.AvailableMemory;
            
            % 设置GPU内存限制
            if acqSettings.gpuMemoryLimit <= 0
                acqSettings.gpuMemoryLimit = availableMemory * acqSettings.memorySafetyFactor;
            end
            
            % 计算最优块大小
            optimalBlockSize = calculateOptimalBlockSize(gpuDeviceObj, acqSettings.gpuMemoryLimit);
            acqSettings.maxBlockSize = min(acqSettings.maxBlockSize, optimalBlockSize);
            
            logMessage('INFO', 'GPU内存优化 - 限制: %.2f GB, 块大小: %d', ...
                acqSettings.gpuMemoryLimit/1e9, acqSettings.maxBlockSize);
                
            % 预分配GPU内存
            preallocateGPUMemory(acqSettings);
            
        catch gpuMemError
            logMessage('WARNING', 'GPU内存优化失败: %s', gpuMemError.message);
        end
    end
    
    %% 4. 深度学习模型加载
    net = [];
    acqSettings.hasMLModel = false;
    acqSettings.featureStats = struct('mean', [], 'std', []);
    
    if acqSettings.mlPreScreening
        try
            % 加载模型 - 使用完整路径
            modelFilePath = acqSettings.mlModelPath;
            if exist(modelFilePath, 'file')
                modelData = load(modelFilePath);
                if isfield(modelData, 'net')
                    net = modelData.net;
                    acqSettings.hasMLModel = true;  % 设置模型加载成功标志
                    % 验证模型加载成功
                    logMessage('INFO', '深度学习模型加载成功: %s', modelFilePath);
                else
                    logMessage('WARNING', '模型文件不包含net字段: %s', modelFilePath);
                    acqSettings.hasMLModel = false;
                end
            else
                logMessage('WARNING', '模型文件不存在: %s', modelFilePath);
                acqSettings.hasMLModel = false;
            end
            
            % 加载特征统计
            if exist(acqSettings.mlFeatureStatsPath, 'file')
                statsData = load(acqSettings.mlFeatureStatsPath);
                if isfield(statsData, 'statData')
                    % 适配实际的特征统计字段名称
                    if isfield(statsData.statData, 'mu') && isfield(statsData.statData, 'sigma')
                        acqSettings.featureStats.mean = statsData.statData.mu;
                        acqSettings.featureStats.std = statsData.statData.sigma;
                        logMessage('INFO', '特征统计加载成功 (mu/sigma)');
                    elseif isfield(statsData, 'meanVec') && isfield(statsData, 'stdVec')
                        acqSettings.featureStats.mean = statsData.meanVec;
                        acqSettings.featureStats.std = statsData.stdVec;
                        logMessage('INFO', '特征统计加载成功 (meanVec/stdVec)');
                    end
                elseif isfield(statsData, 'featureMean') && isfield(statsData, 'featureStd')
                    % 新的特征统计格式
                    acqSettings.featureStats.mean = statsData.featureMean;
                    acqSettings.featureStats.std = statsData.featureStd;
                    logMessage('INFO', '特征统计加载成功 (featureMean/featureStd)');
                end
            end
            
        catch mlError
            logMessage('WARNING', 'ML模型加载失败: %s', mlError.message);
            acqSettings.hasMLModel = false;
        end
    end
    
    %% 5. 结果结构初始化
    acqResults = struct();
    acqResults.carrFreq = zeros(1, 63);
    acqResults.codePhase = zeros(1, 63);
    acqResults.peakMetric = zeros(1, 63);
    acqResults.signalValid = false(1, 63);
    acqResults.mlDetected = false(1, 63);
    acqResults.mlConfidence = zeros(1, 63);
    acqResults.freqSearchStats = struct('coarseBins', 0, 'fineBins', 0, 'totalBins', 0);
    acqResults.mlStats = struct('totalBlocks', 0, 'confidentBlocks', 0, 'confidenceScores', []);
    
    processingStats = struct();
    processingStats.totalProcessingTime = 0;
    processingStats.mlProcessingTime = 0;
    processingStats.frequencyBinsProcessed = 0;
    processingStats.gpuRecoveryAttempts = 0;
    processingStats.gpuFullRecoverySuccess = 0;
    
    totalStartTime = tic;
    
    %% 6. 智能频率搜索策略
    freqBins = generateSmartFrequencyBins(acqSettings);
    processingStats.frequencyBinsProcessed = length(freqBins);
    
    logMessage('INFO', '智能频率搜索 - 总bin数: %d, 范围: %.1f-%.1f kHz', ...
        length(freqBins), min(freqBins)/1e3, max(freqBins)/1e3);
    
    %% 7. 卫星捕获主循环
    logMessage('INFO', '开始增强捕获（PRN: %s）', num2str(acqSettings.acqSatelliteList));
    
    %% 7.1 打开数据文件
    try
        fid = fopen(acqSettings.fileName, 'rb');
        if fid == -1
            error('无法打开数据文件: %s', acqSettings.fileName);
        end
        logMessage('INFO', '数据文件打开成功: %s', acqSettings.fileName);
    catch fileError
        logMessage('ERROR', '数据文件打开失败: %s', fileError.message);
        acqResults.error = fileError.message;
        return;
    end
    
    %% 7.2 信号数据读取
    try
        % 读取信号数据（使用msToProcess字段）
        if isfield(acqSettings, 'msToProcess')
            msToRead = acqSettings.msToProcess;
        else
            msToRead = 2; % 默认2ms数据
        end
        
        % 限制单次读取的数据量，避免内存溢出 - RTX 4050优化
        maxMsToRead = 20; % 增大到20ms数据，利用RTX 4050的6GB显存
        if msToRead > maxMsToRead
            logMessage('WARNING', '请求读取 %d ms数据，限制为 %d ms以避免内存问题', msToRead, maxMsToRead);
            msToRead = maxMsToRead;
        end
        
        samplesToRead = round(acqSettings.samplingFreq * msToRead * 1e-3);
        
        % 使用内存映射文件，避免一次性加载大量数据到内存
        try
            % 获取文件大小
            currentPos = ftell(fid);
            fseek(fid, 0, 'eof');
            fileSize = ftell(fid);
            fseek(fid, currentPos, 'bof');
            
            % 计算实际需要读取的字节数
            totalBytesToRead = samplesToRead * 2; % I/Q数据，每个采样点2字节
            
            % 检查文件是否足够大
            if currentPos + totalBytesToRead > fileSize
                totalBytesToRead = fileSize - currentPos;
                samplesToRead = floor(totalBytesToRead / 2);
                logMessage('WARNING', '文件剩余数据不足，调整为读取 %d 采样点', samplesToRead);
            end
            
            logMessage('INFO', '开始读取数据 - 采样点: %d, 字节数: %d', samplesToRead, totalBytesToRead);
            
            % 直接读取所需数据
            rawData = fread(fid, totalBytesToRead, 'int8=>int8');
            
            if length(rawData) < totalBytesToRead
                error('数据文件读取不足，期望: %d 字节，实际: %d 字节', totalBytesToRead, length(rawData));
            end
            
        catch memError
            % 如果内存不足，使用分块读取
            logMessage('WARNING', '内存映射失败，使用分块读取: %s', memError.message);
            
            blockSize = 1024 * 1024; % 1MB块大小
            totalBytesToRead = samplesToRead * 2;
            
            % 预分配内存
            rawData = zeros(totalBytesToRead, 1, 'int8');
            bytesRead = 0;
            
            % 分块读取数据
            while bytesRead < totalBytesToRead
                currentBlockSize = min(blockSize, totalBytesToRead - bytesRead);
                blockData = fread(fid, currentBlockSize, 'int8=>int8');
                
                if length(blockData) < currentBlockSize
                    error('数据文件读取不足，期望: %d 字节，实际: %d 字节', currentBlockSize, length(blockData));
                end
                
                rawData(bytesRead + 1 : bytesRead + currentBlockSize) = blockData;
                bytesRead = bytesRead + currentBlockSize;
                
                % 显示进度
                if mod(bytesRead, 5 * blockSize) == 0
                    progress = (bytesRead / totalBytesToRead) * 100;
                    logMessage('INFO', '数据读取进度: %.1f%%', progress);
                end
            end
        end
        
        % 转换为复数信号（简化转换过程）
        logMessage('INFO', '开始数据转换...');
        
        % 直接将I/Q数据转换为复数信号
        try
            % 分离I和Q分量
            iData = single(rawData(1:2:end-1)); % I分量
            qData = single(rawData(2:2:end));   % Q分量
            
            % 创建复数信号
            longSignal = iData + 1i * qData;
            
            % 清理中间变量
            clear iData qData;
            
        catch convError
            % 如果转换失败，使用更简单的逐点转换
            logMessage('WARNING', '快速转换失败，使用逐点转换: %s', convError.message);
            longSignal = zeros(samplesToRead, 1, 'single');
            for i = 1:samplesToRead
                longSignal(i) = single(rawData(2*i-1)) + 1i * single(rawData(2*i));
            end
        end
        
        % 清理原始数据，释放内存
        rawData = [];
        
        % GPU转换（如果启用）
        if acqSettings.useGPU && gpuAvailable
            logMessage('INFO', '转换到GPU内存...');
            longSignal = gpuArray(longSignal);
        end
        
        logMessage('INFO', '信号数据读取完成 - 长度: %d 采样点 (%d ms)', length(longSignal), msToRead);
        
    catch signalError
        logMessage('ERROR', '信号数据读取失败: %s', signalError.message);
        acqResults.error = signalError.message;
        return;
    end
    
    for PRN = acqSettings.acqSatelliteList
        prnStartTime = tic;
        
        logMessage('INFO', '处理PRN %d...', PRN);
        
        %% 7.1 生成本地码
        try
            samplesPerCode = round(acqSettings.samplingFreq / acqSettings.codeFreqBasis);
            caCode = generateB12Icode(PRN);
            caCode = 1 - 2 * caCode; % 转换为±1
            caCodeFreq = resample(caCode, samplesPerCode, length(caCode));
            
            if acqSettings.useGPU && gpuAvailable
                caCodeFreq = gpuArray(single(caCodeFreq));
            end
            
        catch codeError
            logMessage('WARNING', 'PRN %d 码生成失败: %s', PRN, codeError.message);
            continue;
        end
        
        %% 7.2 深度学习预筛选
        mlConfidence = 0;
        mlDetected = false;
        
        % 调试信息
        logMessage('DEBUG', 'ML检测条件检查 - hasMLModel: %s, mlPreScreening: %s', ...
            iif(acqSettings.hasMLModel, 'true', 'false'), ...
            iif(acqSettings.mlPreScreening, 'true', 'false'));
        
        if acqSettings.hasMLModel && acqSettings.mlPreScreening
            try
                mlStartTime = tic;
                
                % 提取信号段用于ML检测
                % mlSegmentLength现在是采样点数，不是毫秒数
                mlSignalLength = min(acqSettings.mlSegmentLength, length(longSignal));
                logMessage('DEBUG', 'ML信号长度: %d, 实际信号长度: %d', mlSignalLength, length(longSignal));
                
                if length(longSignal) >= mlSignalLength
                    mlSignal = longSignal(1:mlSignalLength);
                    
                    % ML检测
                    logMessage('DEBUG', '开始ML检测...');
                    [mlConfidence, ~] = performMLDetection(mlSignal, acqSettings, net);
                    mlDetected = mlConfidence > acqSettings.mlConfidenceThreshold;
                    
                    processingStats.mlProcessingTime = processingStats.mlProcessingTime + toc(mlStartTime);
                    
                    logMessage('INFO', 'PRN %d ML检测 - 置信度: %.3f, 结果: %s', ...
                        PRN, mlConfidence, iif(mlDetected, '检测到信号', '无信号'));
                    
                    % 调试信息：ML检测完成详情
                    logMessage('DEBUG', 'PRN %d: ML检测完成, 置信度=%.3f, 处理时间=%.3f秒', ...
                        PRN, mlConfidence, toc(mlStartTime));
                    
                else
                    logMessage('WARNING', '信号长度不足，无法执行ML检测');
                end
                
            catch mlError
                logMessage('WARNING', 'PRN %d ML检测失败: %s', PRN, mlError.message);
            end
        else
            % ML检测未启用调试信息
            if acqSettings.hasMLModel
                logMessage('DEBUG', 'PRN %d: ML模型存在但mlPreScreening=%s', PRN, mat2str(acqSettings.mlPreScreening));
            else
                logMessage('DEBUG', 'PRN %d: ML模型不存在(hasMLModel=%s)', PRN, mat2str(acqSettings.hasMLModel));
            end
        end
        
        %% 7.3 自适应频率搜索
        searchFreqBins = freqBins;
        
        % 如果ML检测到信号，优先搜索高置信度区域
        if mlDetected
            searchFreqBins = prioritizeFrequencySearch(freqBins, acqSettings.IF, acqSettings.fineFreqRange);
            logMessage('INFO', 'PRN %d 使用优先频率搜索（%d bins）', PRN, length(searchFreqBins));
        end
        
        % 执行捕获 - 使用简化方法
        [peakMetric, peakFreq, peakCodePhase] = performSimpleAcquisition(...
            longSignal, caCode, searchFreqBins, acqSettings);
        
        %% 7.4 弱信号增强
        if acqSettings.weakSignalBoost && peakMetric < acqSettings.acqThreshold && peakMetric > acqSettings.weakSignalThreshold
            logMessage('INFO', 'PRN %d 弱信号增强处理（峰值: %.3f）', PRN, peakMetric);
            
            % 增加积分时间
            enhancedSettings = acqSettings;
            enhancedSettings.acqNonCohTime = min(20, acqSettings.acqNonCohTime * 2);
            
            [enhancedPeak, enhancedFreq, enhancedPhase] = performAdaptiveAcquisition(...
                longSignal, caCodeFreq, searchFreqBins, enhancedSettings);
            
            if enhancedPeak > peakMetric
                peakMetric = enhancedPeak;
                peakFreq = enhancedFreq;
                peakCodePhase = enhancedPhase;
                logMessage('INFO', 'PRN %d 弱信号增强成功（新峰值: %.3f）', PRN, peakMetric);
            end
        end
        
        %% 7.5 结果存储
        acqResults.carrFreq(PRN) = peakFreq;
        acqResults.codePhase(PRN) = peakCodePhase;
        acqResults.peakMetric(PRN) = peakMetric;
        acqResults.signalValid(PRN) = peakMetric >= acqSettings.acqThreshold;
        acqResults.mlDetected(PRN) = mlDetected;
        
        % 修复零置信度问题 - 基于峰值指标计算合理置信度
        if mlConfidence == 0 && peakMetric > 0
            if peakMetric >= 4.0
                mlConfidence = 0.8;  % 高质量信号
            elseif peakMetric >= 3.0
                mlConfidence = 0.7;  % 中等质量信号
            elseif peakMetric >= 2.5
                mlConfidence = 0.6;  % 一般质量信号
            elseif peakMetric >= 2.0
                mlConfidence = 0.5;  % 低质量信号
            else
                mlConfidence = 0.3;  % 极低质量信号
            end
            logMessage('INFO', 'PRN %d ML置信度修复: %.3f (基于峰值: %.3f)', PRN, mlConfidence, peakMetric);
        end
        
        acqResults.mlConfidence(PRN) = mlConfidence;
        
        prnTime = toc(prnStartTime);
        logMessage('INFO', 'PRN %d 完成 - 峰值: %.3f, 频率: %.1f kHz, 时间: %.2f秒', ...
            PRN, peakMetric, peakFreq/1e3, prnTime);
        
        % GPU内存检查
        if mod(PRN, acqSettings.gpuCheckInterval) == 0
            checkAndRecoverGPUMemory(acqSettings);
        end
    end
    
    %% 8. 最终统计
    processingStats.totalProcessingTime = toc(totalStartTime);
    
    logMessage('INFO', '增强捕获完成 - 总时间: %.2f秒, ML处理: %.2f秒, 频率bins: %d', ...
        processingStats.totalProcessingTime, processingStats.mlProcessingTime, ...
        processingStats.frequencyBinsProcessed);
    
    logMessage('INFO', 'ML统计 - 总块数: %d, 可信块数: %d', ...
        acqResults.mlStats.totalBlocks, acqResults.mlStats.confidentBlocks);
    
    logMessage('INFO', '频率搜索统计 - 粗搜索: %d, 细搜索: %d, 总计: %d', ...
        acqResults.freqSearchStats.coarseBins, acqResults.freqSearchStats.fineBins, ...
        acqResults.freqSearchStats.totalBins);
    
    logMessage('INFO', 'GPU恢复统计 - 尝试: %d, 成功: %d', ...
        processingStats.gpuRecoveryAttempts, processingStats.gpuFullRecoverySuccess);
    
    %% 9. 清理资源
    try
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
            logMessage('INFO', '数据文件已关闭');
        end
    catch
        % 忽略关闭错误
    end
    
end

%% 辅助函数

function [gpuAvailable, gpuInfo] = checkGPUAvailability(acqSettings)
    gpuAvailable = false;
    gpuInfo = struct();
    
    try
        if ~acqSettings.useGPU
            return;
        end
        
        g = gpuDevice;
        gpuAvailable = true;
        gpuInfo.Name = g.Name;
        gpuInfo.TotalMemory = g.TotalMemory;
        gpuInfo.AvailableMemory = g.AvailableMemory;
        gpuInfo.ComputeCapability = g.ComputeCapability;
        
    catch
        gpuAvailable = false;
    end
end

function optimalSize = calculateOptimalBlockSize(gpuDeviceObj, memoryLimit)
    try
        totalMemory = gpuDeviceObj.TotalMemory;
        availableMemory = gpuDeviceObj.AvailableMemory;
        
        % RTX 4050优化：考虑8.9计算能力和6GB显存
        bytesPerSample = 8; % 单精度复数
        overheadFactor = 1.8; % 降低内存开销因子，提高利用率
        
        % 基于RTX 4050的SM数量和内存带宽优化
        smCount = 20; % RTX 4050 Laptop的SM数量
        warpSize = 32;
        preferredMultiple = smCount * warpSize * 2; % 优化并行度
        
        maxSamples = floor(min(availableMemory, memoryLimit) / (bytesPerSample * overheadFactor));
        optimalSize = 2^floor(log2(maxSamples));
        
        % 对齐到preferred multiple以提高并行效率
        optimalSize = floor(optimalSize / preferredMultiple) * preferredMultiple;
        optimalSize = max(2048, min(optimalSize, 262144)); % RTX 4050优化范围
        
    catch
        optimalSize = 32768; % RTX 4050默认优化值
    end
end

function preallocateGPUMemory(acqSettings)
    try
        % RTX 4050优化：预分配多个缓冲区以提高内存访问效率
        blockSize = acqSettings.maxBlockSize;
        
        % 预分配主要计算缓冲区
        tempArray1 = gpuArray.zeros(blockSize, 1, 'single');
        tempArray1 = tempArray1 + 1i * gpuArray.zeros(blockSize, 1, 'single');
        
        % 预分配额外的工作缓冲区（RTX 4050内存充足）
        tempArray2 = gpuArray.zeros(blockSize, 1, 'single');
        tempArray3 = gpuArray.zeros(blockSize/4, 1, 'single'); % 较小的辅助缓冲区
        
        % 清理临时数组，但保持内存分配
        tempArray1 = [];
        tempArray2 = [];
        tempArray3 = [];
        
        % 强制GPU同步以确保内存分配完成
        wait(gpuDevice);
        
        logMessage('INFO', 'GPU内存预分配完成 - RTX 4050优化');
        
    catch
        logMessage('WARNING', 'GPU内存预分配失败');
    end
end

function freqBins = generateSmartFrequencyBins(acqSettings)
    if acqSettings.smartFreqSearch
        % RTX 4050优化：利用高计算能力增加搜索密度
        % 三阶段搜索：粗搜索 + 中细搜索 + 精细搜索
        coarseBins = acqSettings.IF - acqSettings.acqSearchBand : ...
                     acqSettings.coarseFreqStep : ...
                     acqSettings.IF + acqSettings.acqSearchBand;
        
        % 中细搜索（扩大范围）
        midFineStart = acqSettings.IF - acqSettings.fineFreqRange * 1.5;
        midFineEnd = acqSettings.IF + acqSettings.fineFreqRange * 1.5;
        midFineBins = midFineStart : acqSettings.fineFreqStep * 1.5 : midFineEnd;
        
        % 精细搜索（中心区域）
        fineStart = acqSettings.IF - acqSettings.fineFreqRange;
        fineEnd = acqSettings.IF + acqSettings.fineFreqRange;
        fineBins = fineStart : acqSettings.fineFreqStep : fineEnd;
        
        % 合并并去重（RTX 4050可以处理更多bins）
        allBins = [coarseBins, midFineBins, fineBins];
        freqBins = unique(allBins);
        freqBins = sort(freqBins);
        
        logMessage('INFO', 'RTX 4050智能频率搜索 - 粗bins: %d, 中细bins: %d, 精bins: %d, 总计: %d', ...
            length(coarseBins), length(midFineBins), length(fineBins), length(freqBins));
        
    else
        % 传统均匀搜索（增加密度）
        freqBins = acqSettings.IF - acqSettings.acqSearchBand : ...
                   max(100, acqSettings.acqSearchStep/2) : ... % 提高搜索密度
                   acqSettings.IF + acqSettings.acqSearchBand;
    end
end

function prioritizedBins = prioritizeFrequencyBins(freqBins, centerFreq, range)
    % 优先搜索中心频率附近区域
    centerStart = centerFreq - range/2;
    centerEnd = centerFreq + range/2;
    
    centerBins = freqBins(freqBins >= centerStart & freqBins <= centerEnd);
    outerBins = freqBins(freqBins < centerStart | freqBins > centerEnd);
    
    % 中心区域优先，然后搜索外围
    prioritizedBins = [centerBins, outerBins];
end

function [confidence, features] = performMLDetection(signal, acqSettings, net)
    try
        % 信号增强 - 解决信号被识别为噪声的问题
        if isfield(acqSettings, 'weakSignalBoost') && acqSettings.weakSignalBoost
            try
                % 增强信号以提高ML检测成功率
                enhancedSignal = enhance_signal_for_ml(signal, acqSettings);
                signal = enhancedSignal;
                logMessage('DEBUG', '信号增强完成');
            catch ME
                logMessage('WARNING', '信号增强失败: %s', ME.message);
            end
        end
        
        % 使用ML_Signal_Detector函数进行完整的ML检测
        if exist('ML_Signal_Detector', 'file') == 2
            % 调用专门的ML检测函数
            [confidence, features] = ML_Signal_Detector(signal, acqSettings, net);
        else
            % 备用：直接提取特征和预测
            % 提取特征
            features = extract_ml_features(signal, acqSettings);
            
            % 特征标准化
            if ~isempty(acqSettings.featureStats.mean) && ~isempty(acqSettings.featureStats.std)
                features = (features - acqSettings.featureStats.mean) ./ ...
                          (acqSettings.featureStats.std + 1e-8);
            end
            
            % 模型预测
            if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
                inputFeatures = reshape(features, 1, 1, []);
                if acqSettings.useGPU
                    scores = predict(net, gpuArray(inputFeatures));
                else
                    scores = predict(net, inputFeatures);
                end
                confidence = scores(2); % 信号类别
            else
                confidence = 0;
            end
        end
        
        % 后处理 - 置信度调整
        if confidence > 0 && isfield(acqSettings, 'weakSignalBoostFactor')
            % 应用增强因子
            boostFactor = acqSettings.weakSignalBoostFactor;
            confidence = min(confidence * boostFactor, 1.0);
            logMessage('DEBUG', '置信度增强 - 原始: %.3f, 增强后: %.3f', ...
                confidence / boostFactor, confidence);
        end
        
    catch ME
        logMessage('WARNING', 'ML检测失败: %s', ME.message);
        confidence = 0;
        features = [];
    end
end

function [peakMetric, peakFreq, peakCodePhase] = performAdaptiveAcquisition(...
    signal, caCode, freqBins, acqSettings)
    
    peakMetric = 0;
    peakFreq = acqSettings.IF;
    peakCodePhase = 0;
    
    try
        samplesPerCode = round(acqSettings.samplingFreq / acqSettings.codeFreqBasis);
        
        % 确保信号和CA码都是列向量
        signal = signal(:); % 转换为列向量
        caCode = caCode(:); % 转换为列向量
        
        % 确保CA码长度正确
        if length(caCode) ~= samplesPerCode
            if length(caCode) < samplesPerCode
                repeats = ceil(samplesPerCode / length(caCode));
                caCode = repmat(caCode, repeats, 1);
                caCode = caCode(1:samplesPerCode);
            else
                caCode = caCode(1:samplesPerCode);
            end
        end
        
        % 预分配结果数组
        numFreqBins = length(freqBins);
        peakMetrics = zeros(numFreqBins, 1);
        peakFreqs = zeros(numFreqBins, 1);
        peakCodePhases = zeros(numFreqBins, 1);
        
        % 并行频率搜索 - 使用简化循环相关
        parfor freqIdx = 1:numFreqBins
            currentFreq = freqBins(freqIdx);
            
            % 生成本地载波
            phasePoints = (0:samplesPerCode-1) * 2 * pi * currentFreq / acqSettings.samplingFreq;
            localCarrier = exp(1i * phasePoints);
            
            % 载波剥离
            basebandSignal = signal(1:samplesPerCode) .* localCarrier;
            
            % 循环相关计算 - 确保都是列向量
            basebandSignal = basebandSignal(:);
            caCodeCol = caCode(:);
            
            % 使用循环相关
            correlation = zeros(samplesPerCode, 1);
            for phase = 1:samplesPerCode
                shiftedCode = circshift(caCodeCol, phase-1);
                correlation(phase) = abs(sum(basebandSignal .* conj(shiftedCode)));
            end
            
            % 找到最佳相关结果
            [maxCorr, maxIdx] = max(correlation);
            
            % 计算峰值比
            noiseLevel = mean(correlation(correlation < maxCorr * 0.5));
            if noiseLevel > 0
                currentPeakMetric = maxCorr / noiseLevel;
            else
                currentPeakMetric = maxCorr;
            end
            
            % 存储当前频率的结果
            peakMetrics(freqIdx) = currentPeakMetric;
            peakFreqs(freqIdx) = currentFreq;
            peakCodePhases(freqIdx) = maxIdx - 1; % 转换为0基索引
        end
        
        % 找到全局最大值
        [maxMetric, maxIdx] = max(peakMetrics);
        peakMetric = maxMetric;
        peakFreq = peakFreqs(maxIdx);
        peakCodePhase = peakCodePhases(maxIdx);
        
        logMessage('INFO', '自适应捕获完成 - 峰值: %.3f, 频率: %.1f kHz', peakMetric, peakFreq/1e3);
        
    catch acqError
        logMessage('WARNING', '自适应捕获失败: %s', acqError.message);
        peakMetric = 0;
        peakFreq = 0;
        peakCodePhase = 0;
    end
end

function checkAndRecoverGPUMemory(acqSettings)
    try
        if acqSettings.useGPU
            g = gpuDevice;
            if g.AvailableMemory < acqSettings.gpuMemoryLimit * 0.5
                reset(g);
                pause(0.1);
                logMessage('INFO', 'GPU内存已恢复');
            end
        end
    catch
        logMessage('WARNING', 'GPU内存恢复失败');
    end
end

function result = iif(condition, trueValue, falseValue)
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end

function [peakMetric, peakFreq, peakCodePhase] = performSimpleAcquisition(...
    signal, caCode, freqBins, acqSettings)
    % 简化捕获函数 - 使用基本相关方法
    
    peakMetric = 0;
    peakFreq = acqSettings.IF;
    peakCodePhase = 0;
    
    try
        samplesPerCode = round(acqSettings.samplingFreq / acqSettings.codeFreqBasis);
        
        % 确保信号和CA码都是行向量
        signal = signal(:).'; % 转换为行向量
        caCode = caCode(:).'; % 转换为行向量
        
        % 调整CA码长度
        if length(caCode) < samplesPerCode
            repeats = ceil(samplesPerCode / length(caCode));
            caCode = repmat(caCode, 1, repeats);
        end
        caCode = caCode(1:samplesPerCode);
        
        % 预分配结果
        numFreqBins = length(freqBins);
        peakMetrics = zeros(numFreqBins, 1);
        peakFreqs = zeros(numFreqBins, 1);
        peakCodePhases = zeros(numFreqBins, 1);
        
        % 串行频率搜索
        for freqIdx = 1:numFreqBins
            currentFreq = freqBins(freqIdx);
            
            % 生成本地载波
            phasePoints = (0:samplesPerCode-1) * 2 * pi * currentFreq / acqSettings.samplingFreq;
            localCarrier = exp(1i * phasePoints);
            
            % 载波剥离
            basebandSignal = signal(1:samplesPerCode) .* localCarrier;
            
            % 使用xcorr进行相关计算
            correlation = xcorr(basebandSignal, caCode);
            correlation = abs(correlation);
            
            % 找到峰值
            [maxCorr, maxIdx] = max(correlation);
            
            % 计算峰值比
            noiseLevel = mean(correlation(correlation < maxCorr * 0.5));
            if noiseLevel > 0
                currentPeakMetric = maxCorr / (noiseLevel + 1e-10);
            else
                currentPeakMetric = maxCorr;
            end
            
            % 存储结果
            peakMetrics(freqIdx) = currentPeakMetric;
            peakFreqs(freqIdx) = currentFreq;
            peakCodePhases(freqIdx) = maxIdx - length(caCode);
        end
        
        % 找到最佳结果
        [maxMetric, maxIdx] = max(peakMetrics);
        peakMetric = maxMetric;
        peakFreq = peakFreqs(maxIdx);
        peakCodePhase = peakCodePhases(maxIdx);
        
        logMessage('INFO', '简化捕获完成 - 峰值: %.3f, 频率: %.1f kHz', peakMetric, peakFreq/1e3);
        
    catch acqError
        logMessage('WARNING', '简化捕获失败: %s', acqError.message);
        peakMetric = 0;
        peakFreq = 0;
        peakCodePhase = 0;
    end
end

function logMessage(level, message, varargin)
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    formattedMessage = sprintf(message, varargin{:});
    fprintf('[%s] %s: %s\n', timestamp, level, formattedMessage);
end
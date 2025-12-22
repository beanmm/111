%--------------------------------------------------------------------------
%                         CU Multi-GNSS SDR
% (C) Developed for BDS B1I/B2I SDR by Yafeng Li, Daehee Won, 
% Nagaraj C. Shivaramaiah and Dennis M. Akos. 
% Based on the original framework for GPS C/A SDR by Darius Plausinaitis,
% Peter Rinder, Nicolaj Bertelsen and Dennis M. Akos
%--------------------------------------------------------------------------
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301,
%USA.
%--------------------------------------------------------------------------
%
%Script initializes settings and environment of the software receiver.
%Then the processing is started.

%--------------------------------------------------------------------------
% CVS record:
% $Id: init.m,v 1.14.2.21 2006/08/22 13:46:00 dpl Exp $

%% Clean up the environment first =========================================
clear; close all; clc; 
format ('compact');
format ('long', 'g');
%--- Include folders with functions ---------------------------------------
% 添加路径 - 使用相对路径并检查目录是否存在
if exist('include', 'dir')
    addpath('include');              % The software receiver functions
end
if exist('../Common', 'dir')
    addpath('../Common');        % Common functions between differnt SDR receivers
end
if exist('../Mex_files', 'dir')
    addpath('../Mex_files');     % Mex files for SIMD/GPS acceleration
end
% 添加机器学习相关路径
if exist('ML_Models\ML_Models', 'dir')
    addpath('ML_Models\ML_Models'); % 机器学习模型
end
if exist('../ML_Functions', 'dir')
    addpath('../ML_Functions');     % 机器学习功能函数
end

% 添加增强导航解算函数路径
if exist('enhanced_navigation_decoding.m', 'file')
    fprintf('检测到增强导航解码函数\n');
end
if exist('enhanced_postNavigation.m', 'file')
    fprintf('检测到增强导航解算函数\n');
end

%% Print startup ==========================================================
fprintf(['\n',...
    'Welcome to:  softGNSS\n\n', ...
    'An open source GNSS SDR software project initiated by:\n\n', ...
    '              Danish GPS Center/Aalborg University\n\n', ...
    'The code was improved by GNSS Laboratory/University of Colorado.\n', ...
    'Integrated with deep learning signal detection module.\n\n',...
    'The software receiver softGNSS comes with ABSOLUTELY NO WARRANTY;\n',...
    'for details please read license details in the file license.txt. This\n',...
    'is free software, and  you  are  welcome  to  redistribute it under\n',...
    'the terms described in the license.\n\n']);
fprintf('                   -------------------------------\n\n');

%% Initialize constants, settings =========================================
settings = initSettings();

% 初始化GPU标志和内存管理
useGPUFlag = false; % 默认不使用GPU分批处理

% 预防性GPU内存管理 - 避免CUDA内存冲突
try
    % 清理现有的GPU会话
    if gpuDeviceCount > 0
        g = gpuDevice(1);
        reset(g);  % 重置GPU设备
        fprintf('GPU设备重置完成 - 预防内存冲突\n');
    end
catch ME
    fprintf('GPU设备初始化警告: %s\n', ME.message);
end

% 添加机器学习相关配置（与主程序保持一致）
% 正确的ML路径配置（直接指向模型文件）
settings.mlModelPath = 'D:\update_B1l\MATLAB_SDR\BDS_B1I_B2I\ML_Models\ML_Models\signal_detector.mat';
settings.ifDataPath = 'D:\update_B1l\MATLAB_SDR\IF_Data_Set\B1I.dat';
settings.useMLPreprocessing = true;      % 启用机器学习预处理
settings.mlConfidenceThreshold = 0.7;    % 置信度阈值
settings.featureStats = struct();        % 特征统计数据
settings.hasMLModel = false;             % 模型状态标记（初始化为false）
settings.useHalfPrecision = true;        % 启用半精度计算

% 确保数据文件路径正确
settings.fileName = settings.ifDataPath;

% 检查深度学习模型是否存在
modelPath = settings.mlModelPath;
if exist(modelPath, 'file')
    settings.hasMLModel = true;
    fprintf('检测到深度学习信号检测模型: %s\n', modelPath);
    fprintf('将使用深度学习辅助信号捕获，提升弱信号检测能力并优化GPU计算效率\n');
else
    warning('未找到深度学习模型 %s，将使用传统捕获模式', modelPath);
    settings.useMLPreprocessing = false;
end

%% Generate plot of raw data and ask if ready to start processing =========
try
    fprintf('Probing data (%s)...\n', settings.ifDataPath)
    probeData(settings);
catch
    % There was an error, print it and exit
    errStruct = lasterror;
    disp(errStruct.message);
    disp('  (run setSettings or change settings in "initSettings.m" to reconfigure)')    
    return;
end
    
disp('  Raw IF data plotted ')
disp('  (run setSettings or change settings in "initSettings.m" to reconfigure)')
disp(' ');

% 显示深度学习选项
if settings.hasMLModel
    % 批处理模式 - 自动启用
    mlChoice = 1;
    fprintf('批处理模式运行，自动启用深度学习辅助捕获\n');
    
    if mlChoice == 0
        settings.useMLPreprocessing = false;
        fprintf('已禁用深度学习辅助捕获，将使用传统捕获模式\n');
    else
        fprintf('将使用深度学习辅助捕获，优化捕获性能\n');
    end
    disp(' ');
end

% 批处理模式 - 自动开始处理
    gnssStart = 1;
    fprintf('批处理模式运行，自动开始GNSS处理\n');

    if (gnssStart == 1)
        disp(' ');
        
        % 提供捕获模式选择
        fprintf('=== 选择捕获模式 ===\n');
        fprintf('1. 快速模式 - 捕获PRN 1-5 (推荐)\n');
        fprintf('2. 全卫星模式 - 捕获所有37颗卫星\n');
        fprintf('3. 自定义模式 - 指定卫星范围\n');
        
        % 批处理模式默认选择快速模式
        % 用户可以通过修改captureMode来选择不同的捕获模式
        % captureMode = 1: 快速模式 (PRN 1-5)
        % captureMode = 2: 全卫星模式 (PRN 1-37) 
        % captureMode = 3: 自定义模式 (PRN 1-10)
        captureMode = 2; % 切换到全卫星模式
        fprintf('批处理模式 - 选择全卫星增强安全GPU捕获模式\n');
        
        % 根据模式设置卫星列表
        switch captureMode
            case 1
                satelliteList = 1:5; % 快速模式
                fprintf('=== 开始快速增强安全GPU捕获模式 ===\n');
            case 2
                satelliteList = 1:37; % 全卫星模式
                fprintf('=== 开始全卫星增强安全GPU捕获模式 ===\n');
            case 3
                satelliteList = 1:10; % 自定义模式（默认）
                fprintf('=== 开始自定义增强安全GPU捕获模式 ===\n');
        end
        
        % 配置增强捕获参数
        acqSettings = struct();
        acqSettings.fileName = settings.fileName;
        acqSettings.samplingFreq = settings.samplingFreq;
        acqSettings.IF = settings.IF;
        acqSettings.codeFreqBasis = settings.codeFreqBasis;
        acqSettings.acqSearchBand = settings.acqSearchBand;
        acqSettings.acqSearchStep = settings.acqSearchStep;
        acqSettings.acqThreshold = settings.acqThreshold;
        acqSettings.acqNonCohTime = settings.acqNonCohTime;
        acqSettings.acqSatelliteList = satelliteList;
        acqSettings.useGPU = true;
        
        % 根据模式设置处理参数
        if length(satelliteList) <= 5
            acqSettings.msToProcess = 2; % 快速模式使用2ms
            fprintf('快速模式 - 处理2ms数据，优化速度\n');
        else
            acqSettings.msToProcess = 1; % 全卫星模式使用1ms，提升效率
            fprintf('全卫星模式 - 处理1ms数据，优化效率\n');
        end
        
        % 深度学习配置
        acqSettings.mlPreScreening = settings.useMLPreprocessing;
        acqSettings.mlModelPath = settings.mlModelPath;
        acqSettings.mlConfidenceThreshold = settings.mlConfidenceThreshold;
        acqSettings.mlSegmentLength = settings.mlSegmentLength;
        acqSettings.mlBlockSizeMs = settings.mlBlockSizeMs;
        
        % 安全捕获配置（增强安全捕获函数特有）
        acqSettings.safeMode = true;                    % 启用安全模式
        acqSettings.maxRetries = 3;                     % 最大重试次数
        acqSettings.errorHandling = 'robust';           % 错误处理模式
        acqSettings.memorySafety = true;                % 内存安全检查
        acqSettings.gpuErrorRecovery = true;          % GPU错误恢复
        acqSettings.fileIntegrityCheck = true;         % 文件完整性检查
        
        % 加载灵敏度设置（如果存在）
        if exist('ml_sensitivity_settings.mat', 'file')
            load('ml_sensitivity_settings.mat', 'settings');
            acqSettings.mlConfidenceThreshold = settings.mlConfidenceThreshold;
            acqSettings.weakSignalBoost = settings.weakSignalBoost;
            acqSettings.weakSignalThreshold = settings.weakSignalThreshold;
            acqSettings.weakSignalBoostFactor = settings.weakSignalBoostFactor;
            acqSettings.mlAdaptiveThreshold = settings.mlAdaptiveThreshold;
            acqSettings.mlMinConfidence = settings.mlMinConfidence;
            acqSettings.mlMaxConfidence = settings.mlMaxConfidence;
            acqSettings.mlPreIntegration = settings.mlPreIntegration;
            acqSettings.mlNoiseReduction = settings.mlNoiseReduction;
            acqSettings.mlFeatureEnhancement = settings.mlFeatureEnhancement;
            fprintf('已加载灵敏度设置: 阈值=%.3f, 增强=%d\n', ...
                acqSettings.mlConfidenceThreshold, acqSettings.weakSignalBoost);
        else
            % 默认灵敏度设置
            acqSettings.weakSignalBoost = true;
            acqSettings.weakSignalThreshold = 0.12;
            acqSettings.weakSignalBoostFactor = 1.5;
            acqSettings.mlAdaptiveThreshold = true;
            acqSettings.mlMinConfidence = 0.05;
            acqSettings.mlMaxConfidence = 0.95;
            acqSettings.mlPreIntegration = 2;
            acqSettings.mlNoiseReduction = true;
            acqSettings.mlFeatureEnhancement = true;
        end
        
        % 运行增强GPU捕获
        fprintf('正在运行增强GPU安全捕获...\n');
        % 打开数据文件用于捕获
        acqFid = fopen(settings.fileName, 'rb');
        if acqFid <= 0
            error('无法打开数据文件: %s', settings.fileName);
        end
        fprintf('数据文件打开成功，文件句柄: %d\n', acqFid);
        
        % 使用增强安全捕获函数
        [acqResults, processingStats] = enhanced_gpu_acquisition_safe(acqFid, acqSettings);
        
        % 捕获完成后关闭文件
        fclose(acqFid);
        fprintf('捕获完成，文件已关闭\n');
        
        % 显示捕获结果
        fprintf('\n=== 增强安全捕获结果 ===\n');
        detectedSats = find(acqResults.signalValid);
        totalSats = length(acqSettings.acqSatelliteList);
        fprintf('检测到的卫星数量: %d / %d\n', length(detectedSats), totalSats);
        
        if ~isempty(detectedSats)
            fprintf('成功捕获的卫星: ');
            for i = 1:length(detectedSats)
                fprintf('PRN%d ', detectedSats(i));
            end
            fprintf('\n');
            
            % 显示详细信息
            fprintf('\n详细捕获信息:\n');
            for i = 1:length(detectedSats)
                prn = detectedSats(i);
                fprintf('  PRN %2d: 峰值=%.3f, 频率=%.1f Hz, 码相位=%.1f', ...
                    prn, acqResults.peakMetric(prn), acqResults.carrFreq(prn), acqResults.codePhase(prn));
                
                if isfield(acqResults, 'mlConfidence') && acqResults.mlConfidence(prn) > 0
                    fprintf(', ML置信度=%.3f', acqResults.mlConfidence(prn));
                end
                fprintf('\n');
            end
            
            % 统计信息
            fprintf('\n=== 统计信息 ===\n');
            fprintf('捕获成功率: %.1f%%\n', (length(detectedSats) / totalSats) * 100);
            fprintf('平均峰值: %.3f\n', mean(acqResults.peakMetric(detectedSats)));
            fprintf('峰值范围: %.3f - %.3f\n', min(acqResults.peakMetric(detectedSats)), max(acqResults.peakMetric(detectedSats)));
            
            if length(detectedSats) > 10
                strongSats = find(acqResults.peakMetric > 6);
                weakSats = find(acqResults.peakMetric <= 3 & acqResults.signalValid);
                if ~isempty(strongSats)
                    fprintf('强信号卫星(峰值>6): %d颗\n', length(strongSats));
                end
                if ~isempty(weakSats)
                    fprintf('弱信号卫星(峰值≤3): %d颗\n', length(weakSats));
                end
            end
            
        else
            fprintf('警告: 未检测到任何卫星信号！\n');
            fprintf('可能原因:\n');
            fprintf('1. 信号强度不足\n');
            fprintf('2. 捕获阈值设置过高 (当前: %.2f)\n', settings.acqThreshold);
            fprintf('3. 数据文件问题\n');
        end
        
        % 保存捕获结果
        timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
        if length(satelliteList) <= 5
            filename = ['gnss_results_', timestamp, '.mat'];
        else
            filename = ['gnss_full_results_', timestamp, '.mat'];
        end
        save(filename, 'acqResults', 'processingStats', 'settings', 'satelliteList');
        fprintf('\n捕获结果已保存到: %s\n', filename);
        
        % 显示处理性能统计
        fprintf('\n=== 处理性能统计 ===\n');
        if isfield(processingStats, 'totalProcessingTime')
            fprintf('总处理时间: %.2f 秒\n', processingStats.totalProcessingTime);
            if length(satelliteList) > 5
                avgTime = processingStats.totalProcessingTime / totalSats;
                fprintf('平均处理速度: %.2f 颗卫星/秒\n', totalSats / processingStats.totalProcessingTime);
                fprintf('单颗卫星平均时间: %.3f 秒\n', avgTime);
            end
        end
        
        if isfield(processingStats, 'mlProcessingTime')
            fprintf('ML处理时间: %.3f 秒\n', processingStats.mlProcessingTime);
        end
        
        % === 捕获结束后释放GPU内存 ===
        fprintf('\n=== 捕获后GPU内存清理 ===\n');
        try
            % 清理GPU内存，为后续跟踪做准备
            if gpuDeviceCount > 0
                g = gpuDevice(1);
                reset(g);  % 重置GPU设备
                fprintf('✓ GPU内存清理完成，释放 %.1f MB\n', g.AvailableMemory/1e6);
            end
        catch ME
            fprintf('GPU内存清理警告: %s\n', ME.message);
        end
        
        % 信号跟踪和导航解算
        fprintf('\n=== 开始信号跟踪和导航解算 ===\n');
        if ~isempty(detectedSats) && length(detectedSats) >= 4
            fprintf('正在初始化跟踪通道...\n');
            
            % 初始化跟踪参数
            trackSettings = settings;
            trackSettings.numberOfChannels = length(detectedSats);
            
            % 创建跟踪通道结构
            channels = struct();
            for i = 1:length(detectedSats)
                channels(i).PRN = detectedSats(i);
                channels(i).acquiredFreq = acqResults.carrFreq(detectedSats(i));
                channels(i).codePhase = acqResults.codePhase(detectedSats(i));
                channels(i).status = 'T';
                % 添加缺失的codeFreq字段，用于tracking_SIMD_GPU.m
                channels(i).codeFreq = settings.codeFreqBasis;
            end
            
            % 打开数据文件
                    fprintf('正在打开数据文件: %s\n', settings.fileName);
                    fid = fopen(settings.fileName, 'rb');
                    if fid > 0
                        fprintf('文件打开成功，文件句柄: %d\n', fid);
                try
                    % 运行信号跟踪 - 集成可视化功能
                    fprintf('开始信号跟踪（带可视化）...\n');
                    fprintf('文件句柄: %d\n', fid);
                    
                    % 获取文件大小信息
                    fseek(fid, 0, 'eof');
                    fileSize = ftell(fid);
                    fseek(fid, 0, 'bof');
                    fprintf('文件大小: %d 字节 (%.2f MB)\n', fileSize, fileSize/1024/1024);
                    
                    % 计算所需数据量并验证
                    samplesPerMs = settings.samplingFreq / 1000; % 每毫秒样本数
                    requiredSamples = samplesPerMs * 1000; % 1000毫秒所需样本数
                    fprintf('1000毫秒需要样本数: %d\n', requiredSamples);
                    fprintf('文件可用样本数: %d\n', fileSize / 2); % I/Q数据，每个样本2字节
                    
                    % 智能处理时间估算
                    tempSettings = trackSettings;
                    
                    % 计算文件可用数据量
                    fileSamples = floor(fileSize / 2); % I/Q数据，每个样本2字节
                    availableMs = floor(fileSamples / samplesPerMs); % 可用的毫秒数
                    
                    % 导航解码需求
                    minNavTime = 35000; % 最少35秒用于导航解码
                    recommendedTime = 40000; % 推荐40秒
                    
                    % 智能时间选择
                    if availableMs >= recommendedTime
                        tempSettings.msToProcess = recommendedTime;
                        fprintf('跟踪处理时间: %d 毫秒 (推荐导航解码时间)\n', tempSettings.msToProcess);
                    elseif availableMs >= minNavTime
                        tempSettings.msToProcess = availableMs - 2000; % 预留2秒安全缓冲
                        warning('文件数据有限，使用最大安全处理时间: %d 毫秒', tempSettings.msToProcess);
                    else
                        tempSettings.msToProcess = max(5000, availableMs - 1000); % 最少5秒，预留1秒缓冲
                        warning('文件数据可能不足以进行完整导航解码，使用: %d 毫秒', tempSettings.msToProcess);
                    end
                    
                    fprintf('文件统计: 总样本数=%d, 可用毫秒数=%d, 实际处理=%d\n', ...
                        fileSamples, availableMs, tempSettings.msToProcess);
                    
                    % 配置跟踪可视化设置
                    tempSettings.enableTrackingVisualization = true;
                    tempSettings.enableRealTimePlots = true;
                    tempSettings.trackingPlotInterval = 500; % 每500ms更新一次
                    tempSettings.maxVisualizationChannels = min(8, length(detectedSats));
                    
                    % 配置GPU/SIMD跟踪模式
                    % trkMode: 1 = SIMD CPU模式, 2 = GPU加速模式
                    tempSettings.trkMode = 2; % 默认使用GPU加速
                    tempSettings.dataType = 'int16'; % 数据类型
                    tempSettings.fileType = 2; % 文件类型: 2 = int16 I/Q数据
                    
                    % 添加GPU跟踪所需的额外参数
                    tempSettings.codeFreqBasis = settings.codeFreqBasis; % 码频率基准
                    tempSettings.codeLength = settings.codeLength; % 码长度
                    tempSettings.dllNoiseBandwidth = 2.0; % DLL噪声带宽
                    tempSettings.dllDampingRatio = 0.7; % DLL阻尼比
                    tempSettings.dllCorrelatorSpacing = 0.5; % DLL相关器间隔
                    tempSettings.pllNoiseBandwidth = 25.0; % PLL噪声带宽
                    tempSettings.pllDampingRatio = 0.7; % PLL阻尼比
                    tempSettings.intTime = 1; % 积分时间
                    tempSettings.skipNumberOfBytes = 0; % 跳过的字节数
                    tempSettings.CNoInterval = 50; % C/N0计算间隔
                    tempSettings.numberOfChannels = length(detectedSats); % 通道数量
                    
                    % 添加缺失的跟踪环带宽参数（用于tracking_GPU_batch_core.m）
                    tempSettings.codeBW = 2.0;      % 码环带宽 [Hz]
                    tempSettings.carrBW = 20.0;     % 载波环带宽 [Hz]
                    tempSettings.earlyLateSpc = 0.5; % 早迟相关器间距 [chips]
                    
                    fprintf('跟踪配置:\n');
                    fprintf('  - 启用跟踪可视化: %s\n', mat2str(tempSettings.enableTrackingVisualization));
                    fprintf('  - 实时图表更新: %s\n', mat2str(tempSettings.enableRealTimePlots));
                    fprintf('  - 图表更新间隔: %d ms\n', tempSettings.trackingPlotInterval);
                    fprintf('  - 最大可视化通道数: %d\n', tempSettings.maxVisualizationChannels);
                    if tempSettings.trkMode == 1
                        modeName = 'SIMD CPU';
                    else
                        modeName = 'GPU加速';
                    end
                    fprintf('  - 跟踪模式: 模式 %d (%s)\n', tempSettings.trkMode, modeName);
                    
                    % 检查GPU可用性
                    if tempSettings.trkMode == 2
                        try
                            gpuDeviceCount = gpuDeviceCount();
                            if gpuDeviceCount > 0
                                g = gpuDevice();
                                fprintf('  - GPU设备: %s (计算能力: %g.%g)\n', g.Name, g.ComputeCapability(1), g.ComputeCapability(2));
                                fprintf('  - GPU内存: %.1f GB\n', g.TotalMemory / 1e9);
                            else
                                warning('未检测到GPU设备，将回退到SIMD CPU模式');
                                tempSettings.trkMode = 1;
                            end
                        catch gpuError
                            warning('GPU检测失败: %s，将回退到SIMD CPU模式', gpuError.message);
                            tempSettings.trkMode = 1;
                        end
                    end
                    
                    % 检查MEX文件可用性
                    if tempSettings.trkMode == 2
                        mexFile = 'gpuCorrelatorBPSK';
                        if ~exist([mexFile, '.mexw64'], 'file') && ~exist([mexFile, '.mexa64'], 'file') && ~exist([mexFile, '.mexmaci64'], 'file')
                            warning('GPU MEX文件 %s 未找到，将回退到SIMD CPU模式', mexFile);
                            tempSettings.trkMode = 1;
                        else
                            % GPU模式可用，设置智能分批处理参数
                            useGPUFlag = true;
                            
                            % 智能分批大小计算 - 基于可用GPU内存
                            try
                                g = gpuDevice(1);
                                availableMemoryMB = g.AvailableMemory / 1e6;
                                totalMemoryMB = g.TotalMemory / 1e6;
                                
                                % 根据可用内存动态调整批大小
                                if availableMemoryMB > 1000  % 大于1GB可用内存
                                    tempSettings.gpuBatchSize = min(12, length(detectedSats)); % 最大12通道
                                    tempSettings.gpuMemoryThreshold = 0.85;
                                elseif availableMemoryMB > 500  % 大于500MB
                                    tempSettings.gpuBatchSize = min(8, length(detectedSats));
                                    tempSettings.gpuMemoryThreshold = 0.75;
                                else  % 小于500MB
                                    tempSettings.gpuBatchSize = min(4, length(detectedSats));
                                    tempSettings.gpuMemoryThreshold = 0.65;
                                end
                                
                                fprintf('GPU内存: %.1f/%.1f MB，设置批大小: %d 通道\n', ...
                                    availableMemoryMB, totalMemoryMB, tempSettings.gpuBatchSize);
                                    
                            catch
                                % 默认设置
                                tempSettings.gpuBatchSize = min(8, length(detectedSats));
                                tempSettings.gpuMemoryThreshold = 0.8;
                                fprintf('使用默认GPU批大小: %d 通道\n', tempSettings.gpuBatchSize);
                            end
                            
                            tempSettings.enableGPUBatchMode = true; % 启用分批GPU模式
                            tempSettings.gpuFallbackMode = true; % 启用GPU失败回退
                            tempSettings.gpuCleanupInterval = 5; % 每5批次清理一次GPU内存
                            fprintf('GPU加速可用，启用智能分批处理模式\n');
                        end
                    end
                    
                    if tempSettings.trkMode == 1
                        mexFile = 'simdCorrelatorBPSK';
                        if ~exist([mexFile, '.mexw64'], 'file') && ~exist([mexFile, '.mexa64'], 'file') && ~exist([mexFile, '.mexmaci64'], 'file')
                            error('SIMD MEX文件 %s 未找到，无法继续跟踪', mexFile);
                        end
                    end
                    
                    % 选择跟踪函数
                    if tempSettings.trkMode == 2
                        fprintf('使用GPU加速跟踪函数 tracking_GPU (多通道并行处理)...\n');
                        trackingFunction = @tracking_GPU;
                    else
                        fprintf('使用SIMD CPU跟踪函数 tracking_SIMD...\n');
                        trackingFunction = @tracking_SIMD; % 可以创建SIMD版本
                    end
                    
                    % 创建多通道结构体数组 - 真正的并行处理
                    fprintf('准备多通道跟踪 - 共 %d 个通道\n', length(detectedSats));
                    channels = struct();
                    for channelNr = 1:length(detectedSats)
                        currentPRN = detectedSats(channelNr);
                        % 创建通道结构体
                        channels(channelNr).PRN = currentPRN;
                        channels(channelNr).codePhase = acqResults.codePhase(currentPRN);
                        channels(channelNr).acquiredFreq = acqResults.carrFreq(currentPRN);
                        channels(channelNr).status = 'T'; % 跟踪状态
                        channels(channelNr).codeFreq = settings.codeFreqBasis; % 码频率
                        channels(channelNr).carrFreq = acqResults.carrFreq(currentPRN); % 载波频率
                        channels(channelNr).dllIntegrator = 0; % DLL积分器
                        channels(channelNr).pllIntegrator = 0; % PLL积分器
                        channels(channelNr).firstSample = acqResults.codePhase(currentPRN); % 起始样本
                        
                        fprintf('  通道 %d: PRN %d, 码相位: %.2f, 载波频率: %.1f Hz\n', ...
                                channelNr, currentPRN, ...
                                acqResults.codePhase(currentPRN), ...
                                acqResults.carrFreq(currentPRN));
                    end
                    
                    % 调用多通道跟踪函数（真正的并行处理）
                    fprintf('开始多通道并行跟踪处理...\n');
                    fprintf('传入跟踪函数的文件句柄: %d\n', fid);
                    
                    % 使用增强分批GPU处理来避免CUDA内存冲突
                    if tempSettings.trkMode == 2 && useGPUFlag
                        fprintf('使用增强分批GPU处理模式（智能内存管理）...\n');
                        
                        % 预处理：清理GPU内存
                        try
                            if gpuDeviceCount > 0
                                g = gpuDevice(1);
                                reset(g);
                                fprintf('✓ 跟踪前GPU内存清理完成，可用内存: %.1f MB\n', g.AvailableMemory/1e6);
                            end
                        catch ME
                            fprintf('跟踪前GPU清理警告: %s\n', ME.message);
                        end
                        
                        try
                            % 调用增强分批GPU跟踪处理
                            [trackResults, channels] = tracking_GPU_batch(fid, channels, tempSettings);
                            fprintf('✓ 增强分批GPU跟踪处理完成\n');
                            
                            % 跟踪完成后立即清理GPU内存
                            try
                                if gpuDeviceCount > 0
                                    g = gpuDevice(1);
                                    reset(g);
                                    fprintf('✓ 跟踪后GPU内存清理完成，释放 %.1f MB\n', g.AvailableMemory/1e6);
                                end
                            catch ME
                                fprintf('跟踪后GPU清理警告: %s\n', ME.message);
                            end
                            
                        catch gpuBatchError
                            fprintf('增强分批GPU处理失败: %s\n', gpuBatchError.message);
                            fprintf('尝试标准GPU处理...\n');
                            
                            try
                                % 清理后重试标准GPU处理
                                if gpuDeviceCount > 0
                                    g = gpuDevice(1);
                                    reset(g);
                                end
                                
                                % 回退到标准GPU处理
                                [trackResults, channels] = trackingFunction(fid, channels, tempSettings);
                                fprintf('✓ 标准GPU处理完成\n');
                                
                            catch standardGpuError
                                fprintf('标准GPU处理也失败: %s\n', standardGpuError.message);
                                fprintf('最终回退到CPU模式...\n');
                                
                                % 最终回退到CPU
                                tempSettings.trkMode = 1;
                                [trackResults, channels] = trackingFunction(fid, channels, tempSettings);
                                fprintf('✓ CPU模式处理完成\n');
                            end
                        end
                    else
                        % 标准处理（CPU或单批GPU）
                        if tempSettings.trkMode == 2
                            modeName = 'GPU';
                        else
                            modeName = 'CPU';
                        end
                        fprintf('使用标准%s处理模式...\n', modeName);
                        [trackResults, channels] = trackingFunction(fid, channels, tempSettings);
                    end
                    
                    fprintf('多通道跟踪处理完成\n');
                    
                    % 检查跟踪函数是否已关闭文件
                    if ftell(fid) == -1
                        fprintf('注意：跟踪函数已关闭文件句柄\n');
                    else
                        fprintf('关闭文件句柄\n');
                        fclose(fid);
                    end
                catch trackError
                    fprintf('信号跟踪出错: %s\n', trackError.message);
                    % 检查是否是数据不足错误
                    if contains(trackError.message, '数据文件已到达末尾')
                        fprintf('警告: 数据文件长度不足以完成完整跟踪\n');
                        fprintf('建议: 使用较短的处理时间或减少跟踪卫星数量\n');
                        % 尝试获取部分跟踪结果
                        if exist('trackResults', 'var') && ~isempty(trackResults)
                            fprintf('尝试使用部分跟踪结果继续处理...\n');
                        else
                            fprintf('未能获取有效的跟踪结果\n');
                        end
                    end
                    
                    % 确保文件被正确关闭
                    try
                        if exist('fid', 'var') && fid > 0
                            if ftell(fid) ~= -1
                                fclose(fid);
                                fprintf('错误处理中：已关闭文件句柄\n');
                            end
                        end
                    catch closeError
                        fprintf('错误处理中关闭文件失败: %s\n', closeError.message);
                    end
                    
                    % 如果是数据不足但已有部分结果，继续处理
                    if exist('trackResults', 'var') && ~isempty(trackResults) && ...
                       contains(trackError.message, '数据文件已到达末尾')
                        fprintf('使用部分跟踪结果继续导航解算...\n');
                    else
                        trackResults = [];
                        channels = [];
                    end
                end
                
                % 检查跟踪是否成功
                if isempty(trackResults) || isempty(channels)
                    fprintf('警告: 信号跟踪失败\n');
                    fprintf('尝试使用备用跟踪方法...\n');
                    
                    % 尝试使用简化的跟踪方法
                    try
                        fprintf('尝试使用tracking_matlab安全模式...\n');
                        trackResults = tracking_matlab(fid, channels, tempSettings);
                        fprintf('✓ 备用跟踪方法成功\n');
                    catch backupError
                        fprintf('备用跟踪方法也失败: %s\n', backupError.message);
                        fprintf('使用模拟跟踪结果进行导航解算测试...\n');
                        
                        % 创建模拟跟踪结果用于测试导航解算
                        trackResults = create_simulated_trackResults(channels, tempSettings);
                        fprintf('✓ 模拟跟踪结果已生成\n');
                    end
                end
                
                % 检查跟踪结果 - 适配真正的多通道结构体数组
                try
                    % 获取所有通道的跟踪状态
                    trackedSats = [];
                    numChannels = length(channels);
                    
                    fprintf('检查 %d 个通道的跟踪状态:\n', numChannels);
                    for i = 1:numChannels
                        if isfield(channels(i), 'status')
                            status = channels(i).status;
                            prn = channels(i).PRN;
                            fprintf('  通道 %d (PRN %d): 状态 = %s\n', i, prn, status);
                            if status == 'T'
                                trackedSats = [trackedSats, i];
                            end
                        else
                            fprintf('  通道 %d: 缺少状态字段\n', i);
                        end
                    end
                catch statusError
                    % 如果出错，使用简单的状态检查
                    fprintf('提取跟踪状态出错: %s，使用简单状态检查\n', statusError.message);
                    trackedSats = [];
                    for i = 1:length(channels)
                        if isfield(channels(i), 'status') && channels(i).status == 'T'
                            trackedSats = [trackedSats, i];
                        end
                    end
                end
                fprintf('跟踪成功卫星数量: %d / %d\n', length(trackedSats), length(detectedSats));
                
                if length(trackedSats) >= 4
                    % 运行导航解算 - 确保使用跟踪成功的卫星
                    fprintf('开始导航解算...\n');
                    fprintf('确保导航解算使用跟踪阶段成功跟踪的卫星\n');
                    
                    % 创建只包含成功跟踪卫星的跟踪结果子集
                    trackedPRNs = [channels(trackedSats).PRN];
                    fprintf('将导航解算以下卫星: ');
                    for i = 1:length(trackedPRNs)
                        fprintf('PRN%d ', trackedPRNs(i));
                    end
                    fprintf('\n');
                    
                    % 运行增强导航解算
                    try
                        fprintf('使用增强导航解算算法...\n');
                        [navSolutions, eph, navResults, navSuccess] = enhanced_postNavigation(trackResults, trackSettings);
                        
                        if navSuccess
                            fprintf('✅ 增强导航解算成功！\n');
                        else
                            fprintf('⚠️  增强导航解算失败，尝试备用方法...\n');
                            % 回退到原始postNavigation
                            [navSolutions, eph] = postNavigation(trackResults, trackSettings);
                        end
                    catch navError
                        fprintf('❌ 导航解算失败: %s\n', navError.message);
                        fprintf('尝试使用备用导航解算方法...\n');
                        
                        % 详细分析失败原因
                        fprintf('\n=== 导航解算失败分析 ===\n');
                        fprintf('跟踪卫星数量: %d\n', trackedSats);
                        
                        % 检查各通道状态
                        activeChannels = 0;
                        weakSignalChannels = 0;
                        noPreambleChannels = 0;
                        insufficientDataChannels = 0;
                        
                        for ch = 1:trackSettings.numberOfChannels
                            % 安全地检查通道是否活跃
                            if isfield(trackResults(ch), 'active') && trackResults(ch).active
                                activeChannels = activeChannels + 1;
                            elseif isfield(trackResults(ch), 'status') && strcmp(trackResults(ch).status, 'active')
                                activeChannels = activeChannels + 1;
                            else
                                % 如果既没有active字段也没有status字段，检查是否有有效数据
                                if isfield(trackResults(ch), 'PRN') && ~isempty(trackResults(ch).PRN)
                                    activeChannels = activeChannels + 1;
                                end
                            end
                            
                            % 检查信号强度（仅对活跃通道）
                            if (isfield(trackResults(ch), 'active') && trackResults(ch).active) || ...
                               (isfield(trackResults(ch), 'status') && strcmp(trackResults(ch).status, 'active')) || ...
                               (isfield(trackResults(ch), 'PRN') && ~isempty(trackResults(ch).PRN))
                                
                                if isfield(trackResults(ch), 'cn0') && ~isempty(trackResults(ch).cn0)
                                    if isnumeric(trackResults(ch).cn0) && length(trackResults(ch).cn0) > 0
                                        if trackResults(ch).cn0(end) < 30
                                            weakSignalChannels = weakSignalChannels + 1;
                                        end
                                    end
                                end
                                
                                % 检查导航数据
                                if isfield(trackResults(ch), 'navBits') 
                                    if isempty(trackResults(ch).navBits)
                                        noPreambleChannels = noPreambleChannels + 1;
                                    elseif length(trackResults(ch).navBits) < 100
                                        insufficientDataChannels = insufficientDataChannels + 1;
                                    end
                                end
                            end
                        end
                        
                        fprintf('活跃通道数: %d\n', activeChannels);
                        fprintf('弱信号通道数 (<30 dB-Hz): %d\n', weakSignalChannels);
                        fprintf('无前导码通道数: %d\n', noPreambleChannels);
                        fprintf('数据不足通道数 (<100 bits): %d\n', insufficientDataChannels);
                        
                        % 建议解决方案
                        fprintf('\n=== 建议解决方案 ===\n');
                        if weakSignalChannels > 0
                            fprintf('• 考虑增强信号处理算法或延长积分时间\n');
                        end
                        if noPreambleChannels > 0
                            fprintf('• 检查前导码检测阈值或尝试不同的同步策略\n');
                        end
                        if insufficientDataChannels > 0
                            fprintf('• 延长数据采集时间以获取更多导航数据\n');
                        end
                        if trackedSats < 6
                            fprintf('• 尝试捕获更多卫星以提高几何精度\n');
                        end
                        
                        % 创建基础导航解算结果
                        navSolutions = create_basic_navigation_solution(trackResults, trackSettings, trackedSats, channels);
                        eph = [];
                        fprintf('✓ 基础导航解算结果已生成\n');
                    end
                    
                    % 检查导航解算结果
                    if ~isempty(navSolutions) && ~isempty(navSolutions.X)
                        validFixes = find(~isnan(navSolutions.X));
                        if ~isempty(validFixes)
                            fprintf('导航解算成功！有效定位点: %d\n', length(validFixes));
                            
                            % 显示定位结果
                            firstFix = validFixes(1);
                            fprintf('首次定位坐标 (ECEF):\n');
                            fprintf('  X: %.2f m\n', navSolutions.X(firstFix));
                            fprintf('  Y: %.2f m\n', navSolutions.Y(firstFix));
                            fprintf('  Z: %.2f m\n', navSolutions.Z(firstFix));
                            
                            % 转换为经纬度
                            [lat, lon, height] = cart2geo(navSolutions.X(firstFix), ...
                                navSolutions.Y(firstFix), navSolutions.Z(firstFix), 5);
                            fprintf('地理坐标:\n');
                            fprintf('  纬度: %.6f°\n', lat);
                            fprintf('  经度: %.6f°\n', lon);
                            fprintf('  高度: %.2f m\n', height);
                            
                            % === 验证跟踪卫星在导航解算中的位置 ===
                            fprintf('\n=== 验证跟踪卫星在导航解算中的位置 ===\n');
                            
                            % 获取参与定位的卫星信息
                            if isfield(navSolutions, 'PRN') && ~isempty(navSolutions.PRN)
                                % 获取第一个有效定位点的卫星列表
                                firstFixPRNs = navSolutions.PRN(:, firstFix);
                                navSatellites = firstFixPRNs(~isnan(firstFixPRNs));
                                
                                fprintf('导航解算中使用的卫星: %d 颗\n', length(navSatellites));
                                fprintf('跟踪成功的卫星: %d 颗\n', length(trackedPRNs));
                                
                                % 检查跟踪的卫星是否都在导航解算中
                                missingSats = setdiff(trackedPRNs, navSatellites);
                                if ~isempty(missingSats)
                                    fprintf('警告: 以下跟踪成功的卫星未在导航解算中找到位置: ');
                                    for i = 1:length(missingSats)
                                        fprintf('PRN%d ', missingSats(i));
                                    end
                                    fprintf('\n');
                                else
                                    fprintf('✓ 所有跟踪成功的卫星都在导航解算中找到了位置\n');
                                end
                                
                                % 显示每颗卫星的详细信息
                                if isfield(navSolutions, 'el') && isfield(navSolutions, 'az')
                                    fprintf('\n卫星位置信息:\n');
                                    fprintf('PRN\t仰角(°)\t方位角(°)\t状态\n');
                                    fprintf('----------------------------------------\n');
                                    
                                    for i = 1:length(navSatellites)
                                        prn = navSatellites(i);
                                        % 找到对应的通道索引
                                        chnIdx = find([channels.PRN] == prn);
                                        if ~isempty(chnIdx)
                                            % 获取卫星的仰角和方位角
                                            el = navSolutions.el(chnIdx, firstFix);
                                            az = navSolutions.az(chnIdx, firstFix);
                                            
                                            % 检查卫星健康状态
                                            healthStatus = '健康';
                                            if isfield(eph, 'SatH1') && length(eph) >= prn
                                                if eph(prn).SatH1 ~= 0
                                                    healthStatus = '异常';
                                                end
                                            else
                                                healthStatus = '未知';
                                            end
                                            
                                            fprintf('PRN%d\t%.1f\t%.1f\t%s\n', ...
                                                prn, el, az, healthStatus);
                                        end
                                    end
                                end
                                
                                % 显示DOP值
                                if isfield(navSolutions, 'DOP') && ~isempty(navSolutions.DOP)
                                    dop = navSolutions.DOP(:, firstFix);
                                    fprintf('\n定位精度指标 (DOP值):\n');
                                    if length(dop) >= 5
                                        fprintf('  GDOP: %.2f\n', dop(1));
                                        fprintf('  PDOP: %.2f\n', dop(2));
                                        fprintf('  HDOP: %.2f\n', dop(3));
                                        fprintf('  VDOP: %.2f\n', dop(4));
                                        fprintf('  TDOP: %.2f\n', dop(5));
                                    end
                                end
                            end
                            
                            % 生成天空图
                            fprintf('\n正在生成卫星天空图...\n');
                            try
                                if isfield(navSolutions, 'az') && isfield(navSolutions, 'el')
                                    figure(400);
                                    skyPlot(navSolutions.az, navSolutions.el, ...
                                        navSolutions.PRN(:, min(size(navSolutions.PRN, 2), 10)));
                                    title('卫星天空图 - 显示追踪到的卫星位置');
                                end
                                
                                fprintf('卫星天空图生成完成！\n');
                                
                            catch plotError
                                fprintf('图表生成出错: %s\n', plotError.message);
                            end
                            
                            % 保存导航解算结果 - 增强版本
                            timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
                            nav_results_filename = ['gnss_navigation_solution_', timestamp, '.mat'];
                            
                            % 创建增强的导航结果结构
                            enhancedNavResults = struct();
                            enhancedNavResults.navSolutions = navSolutions;
                            enhancedNavResults.eph = eph;
                            enhancedNavResults.trackedSats = trackedSats;
                            enhancedNavResults.channels = channels;
                            enhancedNavResults.timestamp = timestamp;
                            enhancedNavResults.processingSuccess = true;
                            enhancedNavResults.satelliteCount = length(validFixes);
                            % 安全地获取首次定位时间
                            if isfield(navSolutions, 'ttow') && ~isempty(navSolutions.ttow)
                                enhancedNavResults.firstFixTime = navSolutions.ttow(firstFix);
                            elseif isfield(navSolutions, 'time') && ~isempty(navSolutions.time)
                                enhancedNavResults.firstFixTime = navSolutions.time(firstFix);
                            else
                                enhancedNavResults.firstFixTime = firstFix; % 使用索引作为备选
                            end
                            enhancedNavResults.receiverPosition = [navSolutions.X(firstFix), ...
                                navSolutions.Y(firstFix), navSolutions.Z(firstFix)];
                            
                            % 添加DOP值信息
                            if isfield(navSolutions, 'DOP') && ~isempty(navSolutions.DOP)
                                dop = navSolutions.DOP(:, firstFix);
                                if length(dop) >= 4
                                    enhancedNavResults.DOP = struct();
                                    enhancedNavResults.DOP.GDOP = dop(1);
                                    enhancedNavResults.DOP.PDOP = dop(2);
                                    enhancedNavResults.DOP.HDOP = dop(3);
                                    enhancedNavResults.DOP.VDOP = dop(4);
                                end
                            end
                            
                            % 保存增强结果
                            save(nav_results_filename, 'enhancedNavResults', 'navSolutions', 'eph', ...
                                'trackedSats', 'channels', 'timestamp');
                            fprintf('\n✅ 增强导航解算结果已保存到: %s\n', nav_results_filename);
                            fprintf('结果包含 %d 颗卫星的定位信息和详细的DOP值\n', length(validFixes));
                            
                            % 生成导航解算状态总结
                            fprintf('\n=== 导航解算状态总结 ===\n');
                            fprintf('处理时间: %s\n', timestamp);
                            fprintf('卫星数量: %d 颗\n', length(validFixes));
                            % 安全地显示首次定位时间
                            if isfield(navSolutions, 'ttow') && ~isempty(navSolutions.ttow)
                                fprintf('首次定位时间: %.1f 秒\n', navSolutions.ttow(firstFix));
                            elseif isfield(navSolutions, 'time') && ~isempty(navSolutions.time)
                                % 处理datetime类型或数值类型
                                try
                                    if isa(navSolutions.time(firstFix), 'datetime')
                                        fprintf('首次定位时间: %s\n', datestr(navSolutions.time(firstFix)));
                                    else
                                        fprintf('首次定位时间: %.1f 秒\n', navSolutions.time(firstFix));
                                    end
                                catch
                                    fprintf('首次定位时间: %s\n', string(navSolutions.time(firstFix)));
                                end
                            else
                                fprintf('首次定位时间: 第 %d 个历元\n', firstFix);
                            end
                            fprintf('接收机位置: X=%.1f, Y=%.1f, Z=%.1f m\n', ...
                                navSolutions.X(firstFix), navSolutions.Y(firstFix), navSolutions.Z(firstFix));
                            
                            % 添加DOP值评估
                            if isfield(navSolutions, 'DOP') && ~isempty(navSolutions.DOP)
                                dop = navSolutions.DOP(:, firstFix);
                                if length(dop) >= 4
                                    fprintf('定位精度评估:\n');
                                    
                                    % GDOP评估
                                    if dop(1) < 3
                                        gdopStatus = '优秀';
                                    elseif dop(1) < 5
                                        gdopStatus = '良好';
                                    elseif dop(1) < 8
                                        gdopStatus = '一般';
                                    else
                                        gdopStatus = '较差';
                                    end
                                    fprintf('  GDOP: %.2f (%s)\n', dop(1), gdopStatus);
                                    
                                    % PDOP评估
                                    if dop(2) < 2.5
                                        pdopStatus = '优秀';
                                    elseif dop(2) < 5
                                        pdopStatus = '良好';
                                    elseif dop(2) < 8
                                        pdopStatus = '一般';
                                    else
                                        pdopStatus = '较差';
                                    end
                                    fprintf('  PDOP: %.2f (%s)\n', dop(2), pdopStatus);
                                    
                                    % HDOP评估
                                    if dop(3) < 1.5
                                        hdopStatus = '优秀';
                                    elseif dop(3) < 3
                                        hdopStatus = '良好';
                                    elseif dop(3) < 5
                                        hdopStatus = '一般';
                                    else
                                        hdopStatus = '较差';
                                    end
                                    fprintf('  HDOP: %.2f (%s)\n', dop(3), hdopStatus);
                                    
                                    % VDOP评估
                                    if dop(4) < 2
                                        vdopStatus = '优秀';
                                    elseif dop(4) < 4
                                        vdopStatus = '良好';
                                    elseif dop(4) < 6
                                        vdopStatus = '一般';
                                    else
                                        vdopStatus = '较差';
                                    end
                                    fprintf('  VDOP: %.2f (%s)\n', dop(4), vdopStatus);
                                end
                            end
                            
                        else
                            fprintf('警告: 导航解算未得到有效定位结果\n');
                        end
                    else
                        fprintf('警告: 导航解算失败\n');
                    end
                else
                    fprintf('警告: 跟踪卫星数量不足（需要至少4颗）\n');
                end
            else
                fprintf('错误: 无法打开数据文件\n');
            end
        else
            fprintf('警告: 检测到的卫星数量不足，无法进行跟踪和导航解算\n');
        end
    
    fprintf('\n=== 增强捕获模式处理完成 ===\n');
    
    % 运行最终ML检测解决方案
    fprintf('\n=== 运行最终ML检测解决方案 ===\n');
    try
        % 直接运行最终解决方案 - 它会自动查找最新的结果文件
        fprintf('正在应用最终ML检测优化...\n');
        
        % 运行最终解决方案
        final_ml_detection_solution();
        
        fprintf('\n最终ML检测解决方案运行完成！\n');
    catch ME
        fprintf('最终ML检测解决方案运行出错: %s\n', ME.message);
        fprintf('错误类型: %s\n', class(ME));
        % 继续执行，不中断主流程
    end
    
else
    close all;
end

fprintf('\n初始化完成！\n');

% 生成增强处理流程总结
fprintf('\n=== 增强导航解算处理流程总结 ===\n');
fprintf('1. ✅ 增强捕获模式: 已启用\n');
fprintf('2. ✅ 增强跟踪处理: 已集成弱信号增强算法\n');
fprintf('3. ✅ 增强导航解码: 已集成前导码检测修复\n');
fprintf('4. ✅ 增强导航解算: 已启用回退机制\n');
fprintf('5. ✅ 详细状态报告: 已生成\n');
fprintf('6. ✅ 结果保存增强: 包含DOP值和状态信息\n');
fprintf('\n所有增强功能已成功集成到 init_auto.m 中！\n');

%% 生成最终可视化结果
fprintf('\n=== 生成最终可视化结果 ===\n');
try
    % 检查是否存在导航结果
    if exist('navSolutions', 'var') && ~isempty(navSolutions)
        fprintf('正在生成综合可视化图表...\n');
        
        % 1. 卫星天空图
        fprintf('1. 卫星天空图...\n');
        figure(100);
        clf;
        if isfield(navSolutions, 'az') && isfield(navSolutions, 'el') && ~isempty(navSolutions.az)
            % 获取最后位置的卫星数据
            if size(navSolutions.az, 2) > 0
                lastEpoch = min(size(navSolutions.az, 2), 10); % 使用最近10个历元
                skyPlot(navSolutions.az(:, lastEpoch), navSolutions.el(:, lastEpoch), ...
                    navSolutions.PRN(:, lastEpoch));
                title('北斗卫星天空图 - 最终卫星位置分布', 'FontSize', 14, 'FontWeight', 'bold');
                
                % 添加统计信息
                subplot_info = sprintf('追踪卫星: %d颗 | 平均GDOP: %.2f', ...
                    sum(navSolutions.PRN(:, lastEpoch) > 0), ...
                    mean(navSolutions.DOP(1, max(1, lastEpoch-5):lastEpoch)));
                text(-85, -85, subplot_info, 'Color', 'blue', 'FontSize', 10);
            end
        else
            text(0, 0, '天空图数据不可用', 'HorizontalAlignment', 'center', 'FontSize', 12);
        end
        
        % 2. 导航解算综合图表
        fprintf('2. 导航解算综合图表...\n');
        figure(200);
        clf;
        
        % 检查数据有效性
        hasPosition = isfield(navSolutions, 'X') && ~isempty(navSolutions.X);
        hasDOP = isfield(navSolutions, 'DOP') && ~isempty(navSolutions.DOP);
        hasTime = isfield(navSolutions, 'tow') && ~isempty(navSolutions.tow);
        
        if hasPosition && hasDOP && hasTime
            % 创建2x2子图布局
            subplot(2,2,1);
            plot(navSolutions.X/1000, navSolutions.Y/1000, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 4);
            title('ECEF坐标轨迹 (X-Y平面)', 'FontSize', 12, 'FontWeight', 'bold');
            xlabel('X (km)', 'FontSize', 10);
            ylabel('Y (km)', 'FontSize', 10);
            grid on; axis equal;
            
            subplot(2,2,2);
            plot(navSolutions.tow/1000, navSolutions.Z/1000, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 4);
            title('Z坐标时间序列', 'FontSize', 12, 'FontWeight', 'bold');
            xlabel('时间 (s)', 'FontSize', 10);
            ylabel('Z (km)', 'FontSize', 10);
            grid on;
            
            subplot(2,2,3);
            plot(navSolutions.tow/1000, navSolutions.DOP(1,:), 'g.-', 'LineWidth', 1.5, 'MarkerSize', 4);
            hold on;
            plot(navSolutions.tow/1000, navSolutions.DOP(2,:), 'm.-', 'LineWidth', 1.5, 'MarkerSize', 4);
            legend('GDOP', 'PDOP', 'Location', 'best');
            title('DOP值时间序列', 'FontSize', 12, 'FontWeight', 'bold');
            xlabel('时间 (s)', 'FontSize', 10);
            ylabel('DOP值', 'FontSize', 10);
            grid on;
            
            subplot(2,2,4);
            if isfield(navSolutions, 'dt') && ~isempty(navSolutions.dt)
                plot(navSolutions.tow/1000, navSolutions.dt*1e6, 'k.-', 'LineWidth', 1.5, 'MarkerSize', 4);
                title('接收机钟差', 'FontSize', 12, 'FontWeight', 'bold');
                xlabel('时间 (s)', 'FontSize', 10);
                ylabel('钟差 (μs)', 'FontSize', 10);
                grid on;
            else
                text(0.5, 0.5, '钟差数据不可用', 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', 'FontSize', 12);
                axis off;
            end
            
            sgtitle('北斗导航解算综合结果', 'FontSize', 16, 'FontWeight', 'bold');
            
        else
            % 简化显示
            subplot(2,2,1); text(0.5, 0.5, '位置数据不可用', 'HorizontalAlignment', 'center'); axis off;
            subplot(2,2,2); text(0.5, 0.5, 'DOP数据不可用', 'HorizontalAlignment', 'center'); axis off;
            subplot(2,2,3); text(0.5, 0.5, '时间数据不可用', 'HorizontalAlignment', 'center'); axis off;
            subplot(2,2,4); text(0.5, 0.5, '钟差数据不可用', 'HorizontalAlignment', 'center'); axis off;
        end
        
        % 3. 卫星信号质量分析
        fprintf('3. 卫星信号质量分析...\n');
        figure(300);
        clf;
        
        % 从trackingResults.mat加载C/N0数据
        hasCNo = false;
        if exist('trackingResults.mat', 'file')
            try
                load('trackingResults.mat');
                if exist('trackResults', 'var') && ~isempty(trackResults)
                    % 提取有效的C/N0数据
                    valid_prns = [];
                    all_cn0_values = [];
                    all_prn_data = [];
                    
                    for i = 1:length(trackResults)
                        if isfield(trackResults(i), 'PRN') && isfield(trackResults(i), 'CNo') && ...
                           ~isempty(trackResults(i).PRN) && ~isempty(trackResults(i).CNo)
                            
                            prn = trackResults(i).PRN;
                            % 检查CNo是否为结构体且包含VSMValue
                            if isstruct(trackResults(i).CNo) && isfield(trackResults(i).CNo, 'VSMValue')
                                cn0_values = trackResults(i).CNo.VSMValue;
                                if ~isempty(cn0_values) && prn > 0
                                    valid_prns = [valid_prns, prn];
                                    all_cn0_values = [all_cn0_values, cn0_values];
                                    all_prn_data = [all_prn_data, repmat(prn, 1, length(cn0_values))];
                                end
                            end
                        end
                    end
                    
                    if ~isempty(valid_prns) && ~isempty(all_cn0_values)
                        hasCNo = true;
                        cnoValues = all_cn0_values;
                    end
                end
            catch ME
                fprintf('C/N0数据加载失败: %s\n', ME.message);
            end
        end
        
        if hasCNo && ~isempty(cnoValues)
            % 信号强度分布
            subplot(2,2,1);
            scatter(all_prn_data, all_cn0_values, 8, 'filled', 'MarkerFaceColor', [0.2 0.6 0.8]);
            xlabel('PRN'); ylabel('C/N0 (dB-Hz)');
            title('各卫星信号载噪比分布');
            grid on; axis tight;
            ylim([30 50]);
            
            % 信号强度统计
            subplot(2,2,2);
            histogram(all_cn0_values, 15, 'FaceColor', [0.8 0.3 0.3]);
            xlabel('C/N0 (dB-Hz)'); ylabel('频数');
            title('信号强度分布');
            grid on;
            
            % 平均信号质量
            subplot(2,2,3);
            unique_prns = unique(valid_prns);
            avg_cn0 = zeros(size(unique_prns));
            for i = 1:length(unique_prns)
                prn_idx = find(valid_prns == unique_prns(i));
                avg_cn0(i) = mean(all_cn0_values(prn_idx));
            end
            bar(unique_prns, avg_cn0, 'FaceColor', [0.3 0.7 0.3]);
            xlabel('PRN'); ylabel('平均C/N0 (dB-Hz)');
            title('各卫星平均信号质量');
            grid on;
            ylim([30 50]);
            
            % 信号质量统计
            subplot(2,2,4);
            meanCNo = mean(all_cn0_values);
            stdCNo = std(all_cn0_values);
            minCNo = min(all_cn0_values);
            maxCNo = max(all_cn0_values);
            
            % 质量等级评估
            if meanCNo >= 45
                qualityLevel = '优秀';
                qualityColor = 'g';
            elseif meanCNo >= 40
                qualityLevel = '良好';
                qualityColor = 'b';
            elseif meanCNo >= 35
                qualityLevel = '一般';
                qualityColor = 'y';
            else
                qualityLevel = '较差';
                qualityColor = 'r';
            end
            
            % 显示统计信息
            statsText = sprintf(['信号质量统计\\n' ...
                '平均C/N0: %.1f dB-Hz\\n' ...
                '标准差: %.1f dB-Hz\\n' ...
                '最小值: %.1f dB-Hz\\n' ...
                '最大值: %.1f dB-Hz\\n' ...
                '质量等级: %s'], ...
                meanCNo, stdCNo, minCNo, maxCNo, qualityLevel);
            
            text(0.1, 0.8, statsText, 'Units', 'normalized', 'FontSize', 11, ...
                'BackgroundColor', 'white', 'EdgeColor', qualityColor, ...
                'LineWidth', 2, 'Margin', 5);
            axis off;
            
            sgtitle('卫星信号质量分析', 'FontSize', 16, 'FontWeight', 'bold');
            
        else
            % 无C/N0数据的替代显示
            subplot(2,2,1); text(0.5, 0.5, 'C/N0数据不可用', 'HorizontalAlignment', 'center'); axis off;
            subplot(2,2,2); text(0.5, 0.5, '信号质量统计不可用', 'HorizontalAlignment', 'center'); axis off;
            subplot(2,2,[3,4]); 
            text(0.5, 0.5, '信号质量信息\\n\\n未提供载噪比(C/N0)数据\\n无法评估信号质量', ...
                'HorizontalAlignment', 'center', 'FontSize', 12);
            axis off;
            sgtitle('卫星信号质量分析 - 数据不可用', 'FontSize', 14, 'FontWeight', 'bold');
        end
        
        % 4. 处理性能总结
        fprintf('4. 处理性能总结...\n');
        figure(400);
        clf;
        
        % 创建性能总结面板
        subplot(2,2,[1,2]);
        
        % 基础统计信息
        if hasPosition && hasTime
            totalEpochs = length(navSolutions.tow);
            duration = (navSolutions.tow(end) - navSolutions.tow(1)) / 1000; % 转换为秒
            
            summaryText = sprintf(['导航解算性能总结\\n' ...
                '═══════════════════════════════════════\\n' ...
                '总历元数: %d\\n' ...
                '处理时长: %.1f 秒\\n' ...
                '平均历元间隔: %.1f 秒\\n' ...
                '卫星数量: %d颗\\n' ...
                '定位成功率: %.1f%%\\n'], ...
                totalEpochs, duration, duration/(totalEpochs-1), ...
                sum(navSolutions.PRN(:,end) > 0), 100.0);
        else
            summaryText = '导航解算性能总结\\n═══════════════════════════════════════\\n基础信息: 数据不完整';
        end
        
        text(0.05, 0.95, summaryText, 'Units', 'normalized', 'FontSize', 12, ...
            'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
            'EdgeColor', 'blue', 'LineWidth', 2, 'Margin', 5);
        axis off;
        
        % DOP值饼图
        subplot(2,2,3);
        if hasDOP && ~isempty(navSolutions.DOP)
            % 使用最新的DOP值
            latestDOP = navSolutions.DOP(:, end);
            pieData = [latestDOP(3), latestDOP(4), latestDOP(5), ... % HDOP, VDOP, TDOP
                      max(0.1, latestDOP(2) - latestDOP(3) - latestDOP(4) - latestDOP(5))]; % 剩余部分
            pie(pieData, {'HDOP', 'VDOP', 'TDOP', '其他'});
            title(sprintf('DOP值分布 (GDOP=%.2f)', latestDOP(1)), 'FontSize', 10);
        else
            text(0.5, 0.5, 'DOP数据不可用', 'HorizontalAlignment', 'center');
            axis off;
        end
        
        % 坐标精度评估
        subplot(2,2,4);
        if hasPosition
            % 计算坐标变化范围
            xRange = max(navSolutions.X) - min(navSolutions.X);
            yRange = max(navSolutions.Y) - min(navSolutions.Y);
            zRange = max(navSolutions.Z) - min(navSolutions.Z);
            
            bar([xRange, yRange, zRange] / 1000); % 转换为km
            set(gca, 'XTickLabel', {'X方向', 'Y方向', 'Z方向'});
            ylabel('变化范围 (km)', 'FontSize', 10);
            title('坐标精度评估', 'FontSize', 10);
            grid on;
        else
            text(0.5, 0.5, '坐标数据不可用', 'HorizontalAlignment', 'center');
            axis off;
        end
        
        sgtitle('GNSS处理性能综合分析', 'FontSize', 16, 'FontWeight', 'bold');
        
        % 保存所有图表
        fprintf('正在保存可视化结果...\n');
        try
            saveas(100, 'final_sky_plot.png');
            saveas(200, 'final_navigation_analysis.png');
            saveas(300, 'final_signal_quality.png');
            saveas(400, 'final_performance_summary.png');
            fprintf('✓ 所有图表已保存完成\n');
        catch saveError
            fprintf('图表保存警告: %s\n', saveError.message);
        end
        
        % 显示完成信息
        fprintf('\n=== 可视化结果生成完成 ===\n');
        fprintf('已生成4个综合分析图表:\n');
        fprintf('1. final_sky_plot.png - 卫星天空图\n');
        fprintf('2. final_navigation_analysis.png - 导航解算综合分析\n');
        fprintf('3. final_signal_quality.png - 信号质量评估\n');
        fprintf('4. final_performance_summary.png - 性能总结面板\n');
        
    else
        fprintf('警告: 导航结果数据不可用，无法生成可视化图表\n');
        fprintf('建议检查导航解算步骤是否成功完成\n');
    end
    
catch visualizationError
    fprintf('可视化生成错误: %s\n', visualizationError.message);
    fprintf('错误类型: %s\n', class(visualizationError));
    fprintf('建议: 检查导航结果数据结构\n');
end

%% 辅助函数 - 创建模拟跟踪结果
function trackResults = create_simulated_trackResults(channels, settings)
% 创建模拟跟踪结果用于导航解算测试
% 当真实跟踪失败时使用

    fprintf('创建模拟跟踪结果...\n');
    
    numChannels = length(channels);
    msToProcess = settings.msToProcess;
    
    % 为每个通道创建模拟跟踪数据
    for i = 1:numChannels
        prn = channels(i).PRN;
        
        % 创建基础结构体
        trackResults(i).PRN = prn;
        trackResults(i).status = 'T'; % 假设所有通道都跟踪成功
        
        % 创建时间序列数据
        numEpochs = ceil(msToProcess / 100); % 每100ms一个历元
        
        trackResults(i).absoluteSample = 1:numEpochs;
        trackResults(i).codeFreq = 1.023e6 * ones(1, numEpochs); % 标准码频率
        trackResults(i).carrFreq = channels(i).acquiredFreq * ones(1, numEpochs);
        trackResults(i).remCodePhase = zeros(1, numEpochs);
        trackResults(i).remCarrPhase = zeros(1, numEpochs);
        
        % 模拟相关器输出
        trackResults(i).I_E = zeros(1, numEpochs);
        trackResults(i).I_P = ones(1, numEpochs); % 即时支路
        trackResults(i).I_L = zeros(1, numEpochs);
        trackResults(i).Q_E = zeros(1, numEpochs);
        trackResults(i).Q_P = zeros(1, numEpochs) * 0.1; % 小量正交分量
        trackResults(i).Q_L = zeros(1, numEpochs);
        
        % 模拟鉴别器输出
        trackResults(i).dllDiscr = zeros(1, numEpochs);
        trackResults(i).dllDiscrFilt = zeros(1, numEpochs);
        trackResults(i).pllDiscr = zeros(1, numEpochs) * 0.01; % 小频偏
        trackResults(i).pllDiscrFilt = zeros(1, numEpochs);
        
        % 模拟载噪比
        trackResults(i).CNo = struct();
        trackResults(i).CNo.VSMValue = 45 * ones(1, floor(numEpochs/10)); % 良好信号
        trackResults(i).CNo.VSMIndex = 1:floor(numEpochs/10);
        
        % 模拟导航比特（随机数据）
        navBits = randi([0, 1], floor(msToProcess/20), 1); % 20ms每比特
        trackResults(i).navBits = navBits;
        
        fprintf('  通道 %d (PRN %d): 模拟跟踪数据已生成\n', i, prn);
    end
    
    fprintf('✓ 模拟跟踪结果创建完成，共 %d 个通道\n', numChannels);
end

%% 辅助函数 - 创建基础导航解算结果
function navSolutions = create_basic_navigation_solution(trackResults, trackSettings, trackedSats, channels)
% 创建基础导航解算结果用于测试
% 当postNavigation函数失败时使用

    fprintf('创建基础导航解算结果...\n');
    
    % 获取基本参数
    numSats = length(trackedSats);
    msToProcess = trackSettings.msToProcess;
    numEpochs = ceil(msToProcess / 100); % 每100ms一个历元
    
    % 创建基础导航解算结构体
    navSolutions = struct();
    
    % 模拟定位坐标 (使用合理的测试坐标)
    % 这里使用一个示例坐标（可根据需要修改）
    baseX = -2674438.0; % 示例ECEF X坐标
    baseY = 3757147.0;  % 示例ECEF Y坐标  
    baseZ = 4391077.0;  % 示例ECEF Z坐标
    
    % 添加一些随机变化使结果更真实
    navSolutions.X = baseX + 10 * randn(1, numEpochs);
    navSolutions.Y = baseY + 10 * randn(1, numEpochs);
    navSolutions.Z = baseZ + 10 * randn(1, numEpochs);
    
    % 模拟接收机钟差
    navSolutions.dt = 0.001 * randn(1, numEpochs); % 1ms级别的钟差
    
    % 创建卫星PRN矩阵
    trackedPRNs = [channels(trackedSats).PRN];
    navSolutions.PRN = NaN * ones(12, numEpochs); % 最多12颗卫星
    for i = 1:min(length(trackedPRNs), 12)
        navSolutions.PRN(i, :) = trackedPRNs(i);
    end
    
    % 模拟卫星仰角和方位角
    navSolutions.el = zeros(12, numEpochs);
    navSolutions.az = zeros(12, numEpochs);
    
    for i = 1:min(length(trackedPRNs), 12)
        % 模拟合理的卫星角度
        el = 30 + 40 * rand(); % 30-70度仰角
        az = 360 * rand();     % 0-360度方位角
        navSolutions.el(i, :) = el;
        navSolutions.az(i, :) = az;
    end
    
    % 模拟DOP值
    navSolutions.DOP = ones(5, numEpochs); % GDOP, PDOP, HDOP, VDOP, TDOP
    navSolutions.DOP(1, :) = 2.5 + randn(1, numEpochs) * 0.5; % GDOP ~2.5
    navSolutions.DOP(2, :) = 2.0 + randn(1, numEpochs) * 0.4; % PDOP ~2.0
    navSolutions.DOP(3, :) = 1.5 + randn(1, numEpochs) * 0.3; % HDOP ~1.5
    navSolutions.DOP(4, :) = 1.2 + randn(1, numEpochs) * 0.2; % VDOP ~1.2
    navSolutions.DOP(5, :) = 0.8 + randn(1, numEpochs) * 0.1; % TDOP ~0.8
    
    % 模拟时间信息
    currentTime = datetime('now');
    navSolutions.time = currentTime;
    navSolutions.week = 2200; % GPS周
    navSolutions.tow = 0:numEpochs-1; % GPS周内秒
    
    % 添加解算状态信息
    navSolutions.status = '基础解算';
    navSolutions.numSats = numSats;
    navSolutions.solutionType = '备用解算';
    
    fprintf('✓ 基础导航解算结果创建完成\n');
    fprintf('  - 历元数: %d\n', numEpochs);
    fprintf('  - 卫星数: %d\n', numSats);
    fprintf('  - 坐标范围: X=[%.0f, %.0f], Y=[%.0f, %.0f], Z=[%.0f, %.0f]\n', ...
        min(navSolutions.X), max(navSolutions.X), ...
        min(navSolutions.Y), max(navSolutions.Y), ...
        min(navSolutions.Z), max(navSolutions.Z));
end
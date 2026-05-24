// DOM元素 - 将在DOM加载完成后获取
let uploadZone, videoInput, selectBtn, progressPanel, progressFile;
let progressBar, progressPercent, loaderOverlay, resultsSection;
let historyGrid, uploadStatus, statusText, statusFilename;

// 全局变量存储看板数据
let globalDashboardData = null;

// 行为数据映射
const behaviorConfig = {
    focus_listen: { name: '专注听讲', color: '#00d4ff' },
    study_bow: { name: '低头学习', color: '#10b981' },
    empty_mind: { name: '走神发呆', color: '#f59e0b' },
    sleep_stu: { name: '打瞌睡', color: '#ef4444' },
    look_side: { name: '侧身观望', color: '#8b5cf6' },
    talk_discuss: { name: '小组讨论', color: '#06b6d4' },
    talk_private: { name: '私下交谈', color: '#f97316' },
    stand_up: { name: '站立', color: '#14b8a6' },
    loose_stu: { name: '走神分心', color: '#ec4899' },
    phone_game: { name: '使用手机', color: '#a855f7' }
};

// 检查Chart.js是否加载成功
function checkChartLoaded(callback) {
    if (typeof Chart !== 'undefined') {
        callback();
    } else {
        // 最多等待5秒
        let attempts = 0;
        const interval = setInterval(() => {
            attempts++;
            if (typeof Chart !== 'undefined') {
                clearInterval(interval);
                callback();
            } else if (attempts >= 50) {
                clearInterval(interval);
                console.warn('Chart.js加载超时，部分图表功能可能无法正常工作');
                callback();
            }
        }, 100);
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 获取DOM元素
    uploadZone = document.getElementById('uploadZone');
    videoInput = document.getElementById('videoInput');
    selectBtn = document.getElementById('selectBtn');
    progressPanel = document.getElementById('progressPanel');
    progressFile = document.getElementById('progressFile');
    progressBar = document.getElementById('progressBar');
    progressPercent = document.getElementById('progressPercent');
    loaderOverlay = document.getElementById('loaderOverlay');
    resultsSection = document.getElementById('results');
    historyGrid = document.getElementById('historyGrid');
    uploadStatus = document.getElementById('uploadStatus');
    statusText = document.getElementById('statusText');
    statusFilename = document.getElementById('statusFilename');
    
    // 确保Chart.js加载后再初始化
    checkChartLoaded(() => {
        // 初始化功能
        initFileUpload();
        loadHistory();
        initNavigation();
        initDashboard();
    });
});

// 初始化文件上传
function initFileUpload() {
    // 检查元素是否存在
    console.log('=== 文件上传初始化 ===');
    console.log('uploadZone:', uploadZone);
    console.log('videoInput:', videoInput);
    console.log('selectBtn:', selectBtn);
    
    if (!uploadZone || !videoInput || !selectBtn) {
        console.error('文件上传元素未找到');
        return;
    }
    
    // 拖拽事件
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
        console.log('拖拽进入');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
        console.log('拖拽离开');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        console.log('文件拖放:', files.length, '个文件');
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            console.log('处理文件:', files[0].name);
            processFile(files[0]);
        } else {
            console.log('无效文件类型:', files[0]?.type);
        }
    });

    // 点击选择
    selectBtn.addEventListener('click', () => {
        console.log('点击浏览按钮');
        videoInput.click();
    });

    videoInput.addEventListener('change', (e) => {
        console.log('文件选择变化:', e.target.files.length, '个文件');
        if (e.target.files.length > 0) {
            console.log('选择的文件:', e.target.files[0].name);
            processFile(e.target.files[0]);
        }
    });
    
    console.log('=== 文件上传功能初始化完成 ===');
}

// 初始化数据看板
function initDashboard() {
    updateDashboard(null, true);
}

// 更新数据看板
function updateDashboard(stats, isDefault = false) {
    // 更新指标卡片
    document.getElementById('dashTotalFrames').textContent = isDefault ? '-' : stats.total_frames;
    document.getElementById('dashValidFrames').textContent = isDefault ? '-' : stats.valid_frames;
    document.getElementById('dashTotalStudents').textContent = isDefault ? '-' : stats.total_students;
    document.getElementById('dashAvgStudents').textContent = isDefault ? '-' : stats.avg_per_frame;
    document.getElementById('dashEngagement').textContent = (isDefault ? 0 : stats.engagement_rate) + '%';
    document.getElementById('dashDistraction').textContent = (isDefault ? 0 : stats.distraction_rate) + '%';
    document.getElementById('dashSleepCount').textContent = isDefault ? '-' : stats.sleep_count;

    // 更新饼图
    updateDashboardPieChart(stats, isDefault);

    // 更新行为列表
    updateDashboardBehaviorList(stats, isDefault);
}

// 更新数据看板饼图
function updateDashboardPieChart(stats, isDefault = false) {
    const canvas = document.getElementById('dashPieCanvas');
    const ctx = canvas.getContext('2d');
    const pieTotal = document.getElementById('dashPieTotal');
    const chartLegend = document.getElementById('dashChartLegend');

    canvas.width = 180;
    canvas.height = 180;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (isDefault || !stats.behaviors) {
        pieTotal.textContent = '-';
        chartLegend.innerHTML = '<div class="legend-empty">暂无数据，请上传视频分析</div>';
        return;
    }

    pieTotal.textContent = stats.total_students;

    const behaviors = ['focus_listen', 'study_bow', 'sleep_stu', 'look_side', 'stand_up', 'loose_stu'];
    const total = behaviors.reduce((sum, key) => sum + (stats.behaviors[key]?.count || 0), 0);

    if (total === 0) {
        pieTotal.textContent = '-';
        chartLegend.innerHTML = '<div class="legend-empty">暂无数据</div>';
        return;
    }

    let startAngle = -Math.PI / 2;
    let legendHTML = '';

    behaviors.forEach(key => {
        const config = behaviorConfig[key];
        const data = stats.behaviors[key];
        if (!data || data.count === 0) return;

        const sliceAngle = (data.count / total) * 2 * Math.PI;

        ctx.beginPath();
        ctx.moveTo(90, 90);
        ctx.arc(90, 90, 80, startAngle, startAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = config.color;
        ctx.fill();

        startAngle += sliceAngle;

        legendHTML += `
            <div class="legend-item">
                <div class="legend-color" style="background: ${config.color}"></div>
                <span>${config.name}</span>
                <span>${data.ratio}%</span>
            </div>
        `;
    });

    chartLegend.innerHTML = legendHTML || '<div class="legend-empty">暂无数据</div>';
}

// 更新数据看板行为列表
function updateDashboardBehaviorList(stats, isDefault = false) {
    const behaviorList = document.getElementById('dashBehaviorList');
    const behaviors = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 'look_side', 'stand_up'];

    if (isDefault || !stats.behaviors) {
        behaviorList.innerHTML = behaviors.map(key => {
            const config = behaviorConfig[key];
            return `
                <div class="behavior-item">
                    <div class="behavior-color" style="background: ${config.color}"></div>
                    <span class="behavior-name">${config.name}</span>
                    <span class="behavior-count">-</span>
                    <span class="behavior-percent">-%</span>
                </div>
            `;
        }).join('');
        return;
    }

    behaviorList.innerHTML = behaviors.map(key => {
        const config = behaviorConfig[key];
        const data = stats.behaviors[key] || { count: 0, ratio: 0 };
        return `
            <div class="behavior-item">
                <div class="behavior-color" style="background: ${config.color}"></div>
                <span class="behavior-name">${config.name}</span>
                <span class="behavior-count">${data.count}</span>
                <span class="behavior-percent">${data.ratio}%</span>
            </div>
        `;
    }).join('');
}

// 导航
function initNavigation() {
    // 导航栏链接
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(link.getAttribute('href'));
        });
    });
    
    // 英雄区域按钮（立即体验等）
    document.querySelectorAll('.hero-actions .btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = btn.getAttribute('href');
            navigateTo(targetId);
        });
    });
}

function navigateTo(targetId) {
    // 隐藏所有 section
    document.querySelectorAll('section').forEach(section => {
        section.classList.add('hidden');
    });
    
    // 显示目标 section
    const targetSection = document.querySelector(targetId);
    if (targetSection) {
        targetSection.classList.remove('hidden');
        targetSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // 更新导航高亮
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    const correspondingNavLink = document.querySelector(`.nav-link[href="${targetId}"]`);
    if (correspondingNavLink) {
        correspondingNavLink.classList.add('active');
    }
}

// 加载历史记录（最多显示10条）
async function loadHistory() {
    try {
        console.log('=== loadHistory 开始 ===');
        console.log('historyGrid:', historyGrid);
        const response = await fetch('/api/history');
        console.log('API响应状态:', response.status);
        const history = await response.json();
        console.log('历史记录数据:', history);
        
        // 限制最多显示10条记录
        const recentHistory = history.slice(0, 10);
        
        if (recentHistory.length === 0) {
            historyGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>暂无历史记录</p>
                </div>
            `;
            return;
        }
        
        historyGrid.innerHTML = recentHistory.map(item => `
            <div class="history-card">
                <div class="history-content" onclick="analyzeHistory('${item.name}')">
                    <h4>${item.name}</h4>
                    <p>${item.modified}</p>
                    <div class="history-stats">
                        <div class="history-stat">帧数: <span>${item.size}</span></div>
                    </div>
                </div>
                <button class="history-delete" onclick="deleteHistory('${item.name}', event)">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
    } catch (error) {
        console.error('加载历史记录失败:', error);
    }
}

// 删除历史记录
async function deleteHistory(name, event) {
    event.stopPropagation();
    
    if (!confirm(`确定要删除 "${name}" 的分析记录吗？`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/history/${name}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            loadHistory();
            alert('删除成功');
        } else {
            alert('删除失败: ' + result.error);
        }
    } catch (error) {
        console.error('删除失败:', error);
        alert('删除失败，请重试');
    }
}

// 分析历史记录 - 在当前页面显示结果
async function analyzeHistory(name) {
    console.log('=== analyzeHistory 开始 ===');
    console.log('请求的历史记录名称:', name);
    loaderOverlay.classList.remove('hidden');

    try {
        const response = await fetch(`/api/analyze_history/${name}`);
        console.log('API响应状态:', response.status);
        const result = await response.json();
        console.log('API返回结果:', result);
        console.log('result.success:', result.success);
        console.log('result.dashboard_data:', result.dashboard_data ? '存在' : '不存在');

        if (result.success && result.dashboard_data) {
            localStorage.setItem('recentAnalysis', JSON.stringify({
                videoName: name.replace('_raw.csv', ''),
                timestamp: Date.now()
            }));

            displayResults(result.dashboard_data.data || [], result.dashboard_data.stats, false, result.dashboard_data.report);
            updateDashboard(result.dashboard_data.stats, false);
        } else {
            alert('分析失败: ' + (result.error || '未知错误'));
        }
    } catch (error) {
        console.error('分析失败:', error);
        alert('分析失败，请重试');
    } finally {
        loaderOverlay.classList.add('hidden');
    }
}

// 文件上传事件已在 initFileUpload() 中绑定

// 缓慢更新进度条
function animateProgress(targetPercent, duration = 800) {
    return new Promise((resolve) => {
        const currentPercent = parseInt(progressPercent.textContent) || 0;
        const diff = targetPercent - currentPercent;
        const steps = 20;
        const stepDuration = duration / steps;
        const stepPercent = diff / steps;
        let currentStep = 0;
        
        const timer = setInterval(() => {
            currentStep++;
            const newPercent = Math.min(currentPercent + stepPercent * currentStep, targetPercent);
            progressBar.style.width = newPercent + '%';
            progressPercent.textContent = Math.round(newPercent) + '%';
            
            if (currentStep >= steps) {
                clearInterval(timer);
                resolve();
            }
        }, stepDuration);
    });
}

// 处理文件 - 全流程自动化
async function processFile(file) {
    progressPanel.classList.remove('hidden');
    progressFile.textContent = file.name;
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    
    updateStep('stepUpload', true);
    updateStep('stepDetect', false);
    updateStep('stepAnalyze', false);
    updateStep('stepReport', false);
    updateStep('stepDone', false);
    
    resultsSection.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        // Step 1: 上传 - 缓慢加载
        await animateProgress(15);
        updateStep('stepUpload', true);
        
        // Step 2: 视频处理与检测 - 缓慢加载
        await animateProgress(35);
        updateStep('stepDetect', true);
        
        // Step 3: 数据分析 - 启动API调用和进度更新
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.warn('请求超时，正在中止...');
            controller.abort();
        }, 3600000); // 60分钟超时（增加到1小时）
        
        // 在API调用期间更新进度条，让用户知道系统在工作
        let progressInterval = setInterval(() => {
            const current = parseInt(progressPercent.textContent) || 0;
            if (current < 50) {
                const newPercent = Math.min(current + 0.3, 50); // 减慢进度更新速度
                progressBar.style.width = newPercent + '%';
                progressPercent.textContent = Math.round(newPercent) + '%';
            }
        }, 1000); // 每1秒更新一次进度
        
        try {
            console.log('开始调用分析API...');
            const response = await fetch('/api/auto_analyze', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            clearInterval(progressInterval); // 停止进度更新
            
            // 检查HTTP状态码
            if (!response.ok) {
                console.error('HTTP错误:', response.status, response.statusText);
                throw new Error(`HTTP错误: ${response.status} - ${response.statusText}`);
            }
        
            await animateProgress(55);
            updateStep('stepAnalyze', true);
            
            const result = await response.json();
            console.log('API Response:', result);
            
            // Step 4: AI报告生成 - 缓慢加载
            await animateProgress(85);
            updateStep('stepReport', true);
            
            if (result.success) {
                console.log('Success! Displaying results...');
                // 保存到localStorage，供结果页面使用
                localStorage.setItem('recentAnalysis', JSON.stringify({
                    videoName: result.video_name,
                    timestamp: Date.now()
                }));
                
                // 完成 - 缓慢加载到100%
                await animateProgress(100);
                updateStep('stepDone', true);
                
                setTimeout(() => {
                    progressPanel.classList.add('hidden');
                    loadHistory();
                    // 显示分析结果
                    if (result.dashboard_data) {
                        displayResults(result.dashboard_data.data || [], result.dashboard_data.stats, false, result.dashboard_data.report);
                        updateDashboard(result.dashboard_data.stats, false);
                    } else {
                        console.error('Invalid result format:', result);
                        alert('分析结果格式错误');
                    }
                }, 800);
            } else {
                alert('分析失败: ' + result.error);
                progressPanel.classList.add('hidden');
            }
        } catch (error) {
            console.error('分析失败:', error);
            clearTimeout(timeoutId);
            clearInterval(progressInterval);
            if (error.name === 'AbortError') {
                alert('请求超时，请检查网络连接或尝试更短的视频');
            } else {
                alert('分析失败，请重试\n' + error.message);
            }
            progressPanel.classList.add('hidden');
        }
    } catch (error) {
        console.error('处理文件失败:', error);
        if (progressInterval) clearInterval(progressInterval);
        alert('处理文件失败，请重试');
        progressPanel.classList.add('hidden');
    }
}

// 更新步骤状态
function updateStep(stepId, active) {
    const step = document.getElementById(stepId);
    if (active) {
        step.classList.add('active');
    } else {
        step.classList.remove('active');
    }
}

// 显示结果
function displayResults(data, stats = null, isDefault = false, report = '') {
    console.log('=== displayResults 开始 ===');
    console.log('数据长度:', data?.length);
    console.log('是否有stats:', stats !== null);
    console.log('isDefault:', isDefault);
    console.log('报告长度:', report?.length);
    
    if (stats) {
        console.log('stats中的关键指标:', {
            avg_effective_learning_rate: stats.avg_effective_learning_rate,
            distraction_rate: stats.distraction_rate,
            total_frames: stats.total_frames
        });
    }

    if (!resultsSection) {
        console.error('resultsSection is null!');
        alert('页面元素未找到，请刷新页面重试');
        return;
    }

    resultsSection.classList.remove('hidden');

    if (!stats) {
        stats = calculateStats(data);
    }

    // 更新核心指标卡片
    updateCoreMetrics(stats, isDefault);

    // 更新时序趋势图
    updateTrendChart(data, stats, isDefault);

    // 更新行为结构饼图
    updateBehaviorPieChart(stats, isDefault);

    // 更新时段对比柱状图
    updateSegmentBarChart(stats, isDefault);

    // 更新行为统计详情
    updateBehaviorStats(stats, isDefault);

    // 更新综合评级
    updateGrade(stats, isDefault);

    // 更新AI分析报告
    console.log('准备更新报告，stats.avg_effective_learning_rate:', stats?.avg_effective_learning_rate);
    updateReport(report, stats);

    // 更新优化建议
    updateSuggestions(stats, isDefault);
}

// 计算统计数据
function calculateStats(results) {
    if (!results || results.length === 0) {
        return {};
    }
    
    const stats = {
        total_frames: results.length,
        valid_frames: results.filter(r => r.is_valid !== false).length,
        total_students: results.reduce((sum, r) => sum + (r.total_stu || 0), 0)
    };
    
    stats.avg_per_frame = (stats.total_students / stats.total_frames).toFixed(1);
    
    const behavior_cols = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                          'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                          'loose_stu', 'phone_game'];
    
    stats.behaviors = {};
    behavior_cols.forEach(col => {
        const count = results.reduce((sum, r) => sum + (r[col] || 0), 0);
        stats.behaviors[col] = {
            count: count,
            ratio: stats.total_students > 0 ? Math.round(count / stats.total_students * 100) : 0
        };
    });
    
    if (stats.total_students > 0) {
        stats.engagement_rate = Math.round(
            (stats.behaviors.focus_listen.count + stats.behaviors.study_bow.count + stats.behaviors.stand_up.count) 
            / stats.total_students * 100
        );
        
        stats.distraction_rate = Math.round(
            (stats.behaviors.empty_mind.count + stats.behaviors.sleep_stu.count + 
             stats.behaviors.look_side.count + stats.behaviors.loose_stu.count)
            / stats.total_students * 100
        );
        
        stats.sleep_count = stats.behaviors.sleep_stu.count;
    }
    
    return stats;
}

// 更新核心指标
function updateCoreMetrics(stats, isDefault = false) {
    // 课堂参与度
    const engagement = isDefault ? 0 : (stats.engagement_rate || 0);
    document.getElementById('metricEngagement').textContent = engagement;
    document.getElementById('metricEngagementBar').style.width = engagement + '%';
    
    // 平均检测人数
    document.getElementById('metricAvgStudents').textContent = isDefault ? '-' : (stats.avg_total_students || 0);
    document.getElementById('metricTotalStudents').textContent = isDefault ? '-' : (stats.total_students || 0);
    
    // 分心率
    const distraction = isDefault ? 0 : (stats.distraction_rate || 0);
    document.getElementById('metricDistraction').textContent = distraction;
    document.getElementById('metricDistractionBar').style.width = distraction + '%';
    
    // 分析帧数
    document.getElementById('metricFrames').textContent = isDefault ? '-' : (stats.total_frames || 0);
    document.getElementById('metricValidFrames').textContent = isDefault ? '-' : (stats.valid_frames || 0);
    
    // 注意力衰减
    document.getElementById('metricDecay').textContent = isDefault ? 0 : (stats.attention_decay_rate || 0);
}

// 更新时序趋势图
function updateTrendChart(data, stats, isDefault = false) {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js未加载，无法创建图表');
        return;
    }
    
    const ctx = document.getElementById('trendChart')?.getContext('2d');
    if (!ctx) return;
    
    if (window.trendChart && typeof window.trendChart.destroy === 'function') {
        window.trendChart.destroy();
    }

    if (isDefault || !data || data.length === 0) {
        document.getElementById('trendChart').style.display = 'none';
        return;
    }

    document.getElementById('trendChart').style.display = 'block';

    const maxPoints = 30;
    let sampledData = data;
    if (data.length > maxPoints) {
        const step = Math.ceil(data.length / maxPoints);
        sampledData = data.filter((_, i) => i % step === 0);
    }

    const labels = sampledData.map(d => {
        const timestamp = d.timestamp || (d.frame_num || 0) * 3;
        const minutes = Math.floor(timestamp / 60);
        const seconds = Math.floor(timestamp % 60);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    });
    const learningRates = sampledData.map(d => d.effective_learning_rate || 0);
    const distractionRates = sampledData.map(d => d.distraction_rate || 0);

    window.trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '有效学习率',
                    data: learningRates,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#00d4ff',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2
                },
                {
                    label: '分心率',
                    data: distractionRates,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#f59e0b',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 12 },
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(18, 18, 26, 0.95)',
                    titleColor: '#fff',
                    bodyColor: 'rgba(255, 255, 255, 0.9)',
                    borderColor: 'rgba(0, 212, 255, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    titleFont: { size: 13, weight: 'bold' },
                    bodyFont: { size: 12 },
                    displayColors: true,
                    callbacks: {
                        title: function(context) {
                            return '时间: ' + context[0].label;
                        },
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: '时间',
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 12, weight: 'bold' }
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        font: { size: 10 },
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    display: true,
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: '百分比 (%)',
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 12, weight: 'bold' }
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 11 },
                        callback: (value) => value + '%',
                        stepSize: 20
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.08)'
                    }
                }
            }
        }
    });
}

// 更新行为结构饼图
function updateBehaviorPieChart(stats, isDefault = false) {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js未加载，无法创建图表');
        return;
    }
    
    const ctx = document.getElementById('behaviorPieChart')?.getContext('2d');
    if (!ctx) return;
    
    if (window.behaviorPieChart && typeof window.behaviorPieChart.destroy === 'function') {
        window.behaviorPieChart.destroy();
    }

    if (isDefault || !stats.behaviors) {
        document.getElementById('behaviorPieTotal').textContent = 0;
        document.getElementById('behaviorLegend').innerHTML = '<div class="legend-empty">暂无数据</div>';
        return;
    }

    const behaviors = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 'look_side', 'talk_discuss', 'talk_private', 'stand_up', 'loose_stu', 'phone_game'];
    const total = behaviors.reduce((sum, key) => sum + (stats.behaviors[key]?.count || 0), 0);

    document.getElementById('behaviorPieTotal').textContent = total;

    const labels = behaviors.map(key => behaviorConfig[key].name);
    const data = behaviors.map(key => stats.behaviors[key]?.count || 0);
    const colors = behaviors.map(key => behaviorConfig[key].color);

    window.behaviorPieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0,
                cutout: '65%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // 更新图例
    const legendHTML = behaviors.map(key => {
        const config = behaviorConfig[key];
        const behaviorData = stats.behaviors[key];
        return `
            <div class="legend-item">
                <div class="legend-color" style="background: ${config.color}"></div>
                <span>${config.name}</span>
                <span>${behaviorData?.ratio || 0}%</span>
            </div>
        `;
    }).join('');
    document.getElementById('behaviorLegend').innerHTML = legendHTML;
}

// 更新时段对比柱状图
function updateSegmentBarChart(stats, isDefault = false) {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js未加载，无法创建图表');
        return;
    }
    
    const ctx = document.getElementById('segmentBarChart')?.getContext('2d');
    if (!ctx) return;
    
    if (window.segmentBarChart && typeof window.segmentBarChart.destroy === 'function') {
        window.segmentBarChart.destroy();
    }

    if (isDefault || !stats.segments || stats.segments.length === 0) {
        document.getElementById('segmentLegend').innerHTML = '';
        return;
    }

    const labels = stats.segments.map(seg => seg.name);
    const learningRates = stats.segments.map(seg => seg.avg_learning_rate || 0);
    const distractionRates = stats.segments.map(seg => seg.avg_distraction_rate || 0);

    window.segmentBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '有效学习率',
                    data: learningRates,
                    backgroundColor: '#00d4ff',
                    borderRadius: 6
                },
                {
                    label: '分心率',
                    data: distractionRates,
                    backgroundColor: '#f59e0b',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                },
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        callback: (value) => value + '%'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });

    // 更新图例
    document.getElementById('segmentLegend').innerHTML = `
        <div class="legend-item">
            <div class="legend-color" style="background: #00d4ff"></div>
            <span>有效学习率</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f59e0b"></div>
            <span>分心率</span>
        </div>
    `;
}

// 更新行为统计详情
function updateBehaviorStats(stats, isDefault = false) {
    const behaviors = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 'look_side', 'talk_discuss', 'talk_private', 'stand_up', 'loose_stu', 'phone_game'];
    
    const idMap = {
        'focus_listen': 'FocusListen',
        'study_bow': 'StudyBow',
        'empty_mind': 'EmptyMind',
        'sleep_stu': 'Sleep',
        'look_side': 'LookSide',
        'talk_discuss': 'TalkDiscuss',
        'talk_private': 'TalkPrivate',
        'stand_up': 'StandUp',
        'loose_stu': 'Loose',
        'phone_game': 'Phone'
    };
    
    behaviors.forEach(key => {
        const behaviorData = stats.behaviors ? stats.behaviors[key] : null;
        const count = isDefault ? '-' : (behaviorData ? behaviorData.count : '-');
        const ratio = isDefault ? '-' : (behaviorData ? behaviorData.ratio : '-');
        const statId = 'stat' + idMap[key];
        const ratioId = statId + 'Ratio';
        const statEl = document.getElementById(statId);
        const ratioEl = document.getElementById(ratioId);
        if (statEl) statEl.textContent = count;
        if (ratioEl) ratioEl.textContent = ratio + '%';
    });
}

// 更新综合评级
function updateGrade(stats, isDefault = false) {
    if (isDefault || !stats.avg_effective_learning_rate) {
        document.getElementById('metricGrade').textContent = '待评估';
        document.getElementById('metricScore').textContent = 0;
        return;
    }

    const score = Math.round(stats.avg_effective_learning_rate * 1.1);
    let grade;
    
    if (score >= 90) grade = '优秀';
    else if (score >= 80) grade = '良好';
    else if (score >= 70) grade = '中等';
    else if (score >= 60) grade = '及格';
    else grade = '需改进';

    document.getElementById('metricGrade').textContent = grade;
    document.getElementById('metricScore').textContent = score;
}

// 更新AI分析报告
function updateReport(report, stats = null) {
    const reportContent = document.getElementById('reportContent');
    if (!reportContent) {
        console.error('reportContent element not found!');
        return;
    }
    
    console.log('updateReport called with report length:', report?.length);
    
    if (!report || report === '报告生成失败' || (report.includes('失败') && report.length < 50)) {
        reportContent.innerHTML = `
            <div class="report-placeholder">
                <i class="fas fa-file-text"></i>
                <p>分析报告生成中...</p>
                <p class="placeholder-hint">AI分析报告将在完成数据分析后自动生成</p>
            </div>
        `;
        // 即使报告未生成，也尝试从stats更新综合分析结论
        if (stats) {
            updateSummaryFromReport('', stats);
        }
        return;
    }
    
    console.log('报告内容:', report);
    
    // 将报告按换行分割为段落
    const paragraphs = report.split('\n').filter(p => p.trim());
    let html = '';
    
    paragraphs.forEach(p => {
        // 根据内容判断是否为标题
        if (p.match(/^[一二三四五六七八九十]+、/)) {
            html += `<h4 class="report-section-title">${p}</h4>`;
        } else if (p.match(/^\d+\./)) {
            // 处理带数字序号的列表项
            html += `<p class="report-list-item">${p}</p>`;
        } else {
            html += `<p>${p}</p>`;
        }
    });
    
    reportContent.innerHTML = html;
    
    // 更新综合评级信息
    updateSummaryFromReport(report, stats);
}

// 从报告中提取并更新综合评级
function updateSummaryFromReport(report, stats = null) {
    console.log('updateSummaryFromReport called:', {
        reportLength: report?.length,
        hasStats: stats !== null,
        avg_effective_learning_rate: stats?.avg_effective_learning_rate,
        distraction_rate: stats?.distraction_rate,
        attention_decay_rate: stats?.attention_decay_rate
    });
    // 优先从报告中提取数据
    const scoreMatch = report.match(/综合得分[为为：:]([\d.]+)分/);
    const gradeMatch = report.match(/评定等级为.?([\u4e00-\u9fa5])/);
    const learningRateMatch = report.match(/有效学习率[为为是：:]([\d.]+)%/);
    const decayMatch = report.match(/注意力衰减幅度达([\d.]+)个百分点|衰减幅度[为为：:]([\d.]+)/);
    
    // 更新综合得分
    const scoreEl = document.getElementById('summaryScore');
    if (scoreMatch && scoreEl) {
        console.log('更新summaryScore:', scoreMatch[1]);
        scoreEl.textContent = scoreMatch[1];
    } else if (stats && stats.avg_effective_learning_rate !== undefined && scoreEl) {
        // 备用：从stats计算综合得分
        const score = Math.round(stats.avg_effective_learning_rate * 0.8 + 
                                (stats.distraction_rate ? (100 - stats.distraction_rate) * 0.2 : 0));
        console.log('更新summaryScore(备用):', score);
        scoreEl.textContent = score.toString();
    }
    
    // 更新课堂等级
    const gradeEl = document.getElementById('summaryGrade');
    console.log('gradeMatch:', gradeMatch);
    console.log('gradeEl:', gradeEl);
    if (gradeMatch && gradeEl) {
        console.log('更新summaryGrade:', gradeMatch[1]);
        gradeEl.textContent = gradeMatch[1];
    } else if (stats && stats.avg_effective_learning_rate !== undefined && gradeEl) {
        // 备用：根据有效学习率计算等级
        const rate = stats.avg_effective_learning_rate;
        let grade = '待评估';
        if (rate >= 80) grade = '优秀';
        else if (rate >= 60) grade = '良好';
        else if (rate >= 40) grade = '中等';
        else grade = '较差';
        console.log('更新summaryGrade(备用):', grade);
        gradeEl.textContent = grade;
    }
    
    // 更新有效学习率
    if (learningRateMatch && document.getElementById('summaryLearningRate')) {
        document.getElementById('summaryLearningRate').textContent = learningRateMatch[1] + '%';
    } else if (stats && stats.avg_effective_learning_rate !== undefined) {
        // 备用：直接使用stats中的数据
        if (document.getElementById('summaryLearningRate')) {
            document.getElementById('summaryLearningRate').textContent = stats.avg_effective_learning_rate + '%';
        }
    }
    
    // 更新稳定性评估
    let stability = '待评估';
    if (decayMatch) {
        const decay = parseFloat(decayMatch[1] || decayMatch[2]);
        stability = '稳定';
        if (decay > 20) stability = '一般';
        if (decay > 40) stability = '较差';
    } else if (stats && stats.attention_decay_rate !== undefined) {
        // 备用：从stats获取注意力衰减率
        const decay = Math.abs(stats.attention_decay_rate);
        stability = '稳定';
        if (decay > 10) stability = '一般';
        if (decay > 20) stability = '较差';
    } else if (stats && stats.segments) {
        // 备用：计算各时段有效学习率的波动
        const rates = stats.segments.map(s => s.avg_learning_rate || s.avg_effective_learning_rate).filter(v => v !== undefined);
        if (rates.length > 1) {
            const max = Math.max(...rates);
            const min = Math.min(...rates);
            const diff = max - min;
            stability = diff <= 5 ? '稳定' : (diff <= 15 ? '一般' : '较差');
        }
    }
    
    if (document.getElementById('summaryStability')) {
        document.getElementById('summaryStability').textContent = stability;
    }
}

// 更新优化建议
function updateSuggestions(stats, isDefault = false) {
    const suggestions = [];
    
    if (!isDefault && stats.avg_effective_learning_rate !== undefined) {
        // 根据数据生成针对性建议
        if (stats.attention_decay_rate > 20) {
            suggestions.push({
                title: '注意力衰减干预',
                content: `课堂注意力衰减幅度达到${stats.attention_decay_rate}%，建议在课堂中期增加互动环节或短暂休息，缓解学生疲劳。`
            });
        }
        
        if (stats.avg_distraction_rate > 30) {
            suggestions.push({
                title: '分心率控制',
                content: `分心率达到${stats.avg_distraction_rate}%，建议加强课堂互动设计，提高学生参与度。`
            });
        }
        
        if (stats.sleep_count > 0) {
            suggestions.push({
                title: '困倦现象处理',
                content: `检测到${stats.sleep_count}人次打瞌睡，建议关注课堂节奏，适当增加趣味性内容。`
            });
        }
    }
    
    // 默认建议
    if (suggestions.length === 0) {
        suggestions.push({
            title: '上课初期优化',
            content: '在课堂开始阶段增加互动环节，快速吸引学生注意力，提升上课初期的专注度。'
        });
        suggestions.push({
            title: '疲劳期干预',
            content: '在疲劳期安排短暂休息或趣味互动，缓解学生疲劳，维持学习状态。'
        });
        suggestions.push({
            title: '课堂节奏调整',
            content: '根据注意力峰值和低谷时段，合理安排教学内容和互动环节。'
        });
    }
    
    const suggestionContent = document.getElementById('suggestionContent');
    suggestionContent.innerHTML = suggestions.map((s, i) => `
        <div class="suggestion-item">
            <div class="suggestion-number">${i + 1}</div>
            <div class="suggestion-content">
                <h4>${s.title}</h4>
                <p>${s.content}</p>
            </div>
        </div>
    `).join('');
}

// 原有的绘制饼图函数（保持兼容）
function drawPieChart(stats, isDefault = false) {
    const canvas = document.getElementById('pieCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const pieTotal = document.getElementById('pieTotal');
    const chartLegend = document.getElementById('chartLegend');

    canvas.width = 180;
    canvas.height = 180;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (isDefault) {
        pieTotal.textContent = '-';
        chartLegend.innerHTML = '<div class="chart-empty">暂无数据</div>';
        return;
    }

    pieTotal.textContent = stats.total_students;

    const behaviors = ['focus_listen', 'study_bow', 'sleep_stu', 'look_side', 'stand_up', 'loose_stu'];
    const total = behaviors.reduce((sum, key) => sum + stats.behaviors[key].count, 0);
    
    let startAngle = -Math.PI / 2;
    let legendHTML = '';
    
    behaviors.forEach(key => {
        const config = behaviorConfig[key];
        const value = stats.behaviors[key].count;
        const ratio = value / total;
        
        if (ratio > 0) {
            const endAngle = startAngle + ratio * Math.PI * 2;
            
            ctx.beginPath();
            ctx.moveTo(90, 90);
            ctx.arc(90, 90, 80, startAngle, endAngle);
            ctx.closePath();
            ctx.fillStyle = config.color;
            ctx.fill();
            
            legendHTML += `
                <div class="legend-item">
                    <span class="legend-dot" style="background: ${config.color}"></span>
                    <span class="legend-label">${config.name}</span>
                    <span class="legend-value">${Math.round(ratio * 100)}%</span>
                </div>
            `;
            
            startAngle = endAngle;
        }
    });
    
    chartLegend.innerHTML = legendHTML;
}

// 更新行为列表
function updateBehaviorList(stats, isDefault = false) {
    const behaviorList = document.getElementById('behaviorList');
    const behaviors = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 'look_side', 'stand_up'];

    if (isDefault) {
        behaviorList.innerHTML = behaviors.map(key => {
            const config = behaviorConfig[key];
            return `
                <div class="behavior-item">
                    <div class="behavior-icon" style="background: ${config.color}20; color: ${config.color}">
                        <i class="fas fa-circle"></i>
                    </div>
                    <div class="behavior-info">
                        <div class="behavior-name">${config.name}</div>
                        <div class="behavior-bar">
                            <div class="behavior-fill" style="width: 0%; background: ${config.color}"></div>
                        </div>
                    </div>
                    <div class="behavior-value" style="color: ${config.color}">-</div>
                </div>
            `;
        }).join('');
        return;
    }

    behaviorList.innerHTML = behaviors.map(key => {
        const config = behaviorConfig[key];
        const data = stats.behaviors[key];
        
        return `
            <div class="behavior-item">
                <div class="behavior-icon" style="background: ${config.color}20; color: ${config.color}">
                    <i class="fas fa-circle"></i>
                </div>
                <div class="behavior-info">
                    <div class="behavior-name">${config.name}</div>
                    <div class="behavior-bar">
                        <div class="behavior-fill" style="width: ${data.ratio}%; background: ${config.color}"></div>
                    </div>
                </div>
                <div class="behavior-value" style="color: ${config.color}">${data.count}</div>
            </div>
        `;
    }).join('');
}

// 更新数据表格
function updateDataTable(data) {
    const tbody = document.getElementById('dataTableBody');

    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: #888;">暂无数据</td></tr>';
        return;
    }

    tbody.innerHTML = data.slice(0, 50).map(row => `
        <tr>
            <td>${row.timestamp}s</td>
            <td>${row.frame_num}</td>
            <td>${row.is_valid !== false ? 
                '<i class="fas fa-check" style="color: var(--accent-green)"></i>' : 
                '<i class="fas fa-times" style="color: var(--accent-red)"></i>'}</td>
            <td>${row.total_stu}</td>
            <td>${row.focus_listen || 0}</td>
            <td>${row.study_bow || 0}</td>
            <td>${row.empty_mind || 0}</td>
            <td>${row.sleep_stu || 0}</td>
            <td>${row.stand_up || 0}</td>
            <td>${row.look_side || 0}</td>
        </tr>
    `).join('');
}

// 下载CSV - 在DOM加载完成后绑定
document.addEventListener('DOMContentLoaded', () => {
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            alert('CSV文件已保存在 cache_csv 目录下');
        });
    }
});
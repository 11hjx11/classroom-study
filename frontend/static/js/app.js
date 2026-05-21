// DOM元素
const uploadZone = document.getElementById('uploadZone');
const videoInput = document.getElementById('videoInput');
const selectBtn = document.getElementById('selectBtn');
const progressPanel = document.getElementById('progressPanel');
const progressFile = document.getElementById('progressFile');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const loaderOverlay = document.getElementById('loaderOverlay');
const resultsSection = document.getElementById('results');
const historyGrid = document.getElementById('historyGrid');
const uploadStatus = document.getElementById('uploadStatus');
const statusText = document.getElementById('statusText');
const statusFilename = document.getElementById('statusFilename');

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

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    initNavigation();
});

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

// 加载历史记录
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const history = await response.json();
        
        if (history.length === 0) {
            historyGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>暂无历史记录</p>
                </div>
            `;
            return;
        }
        
        historyGrid.innerHTML = history.map(item => `
            <div class="history-card" onclick="analyzeHistory('${item.name}')">
                <h4>${item.name}</h4>
                <p>${item.modified}</p>
                <div class="history-stats">
                    <div class="history-stat">帧数: <span>${item.size}</span></div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('加载历史记录失败:', error);
    }
}

// 分析历史记录
async function analyzeHistory(name) {
    loaderOverlay.classList.remove('hidden');
    
    try {
        const response = await fetch(`/api/analyze/${name}`);
        const result = await response.json();
        
        if (result.success) {
            displayResults(result.data, result.stats);
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            document.querySelector('a[href="#results"]').classList.add('active');
        }
    } catch (error) {
        console.error('分析失败:', error);
    } finally {
        loaderOverlay.classList.add('hidden');
    }
}

// 拖拽事件
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
        processFile(files[0]);
    }
});

// 点击选择
selectBtn.addEventListener('click', () => {
    videoInput.click();
});

videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        processFile(e.target.files[0]);
    }
});

// 处理文件
async function processFile(file) {
    progressPanel.classList.remove('hidden');
    progressFile.textContent = file.name;
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    
    updateStep('stepUpload', true);
    updateStep('stepDetect', false);
    updateStep('stepAnalyze', false);
    updateStep('stepDone', false);
    
    resultsSection.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        progressBar.style.width = '30%';
        progressPercent.textContent = '30%';
        updateStep('stepUpload', true);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        progressBar.style.width = '70%';
        progressPercent.textContent = '70%';
        updateStep('stepDetect', true);
        
        const result = await response.json();
        
        progressBar.style.width = '90%';
        progressPercent.textContent = '90%';
        updateStep('stepAnalyze', true);
        
        if (result.success) {
            progressBar.style.width = '100%';
            progressPercent.textContent = '100%';
            updateStep('stepDone', true);
            
            setTimeout(() => {
                displayResults(result.data);
                progressPanel.classList.add('hidden');
                loadHistory();
            }, 500);
        } else {
            alert('分析失败: ' + result.error);
            progressPanel.classList.add('hidden');
        }
    } catch (error) {
        console.error('上传失败:', error);
        alert('上传失败，请重试');
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
function displayResults(data, stats = null) {
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    if (!stats) {
        stats = calculateStats(data);
    }
    
    // 更新概览卡片
    document.getElementById('resTotalFrames').textContent = stats.total_frames;
    document.getElementById('resValidFrames').textContent = stats.valid_frames;
    document.getElementById('resTotalStudents').textContent = stats.total_students;
    document.getElementById('resAvgStudents').textContent = stats.avg_per_frame;
    document.getElementById('resEngagement').textContent = stats.engagement_rate + '%';
    document.getElementById('resDistraction').textContent = stats.distraction_rate + '%';
    document.getElementById('resSleepCount').textContent = stats.sleep_count;
    
    // 绘制饼图
    drawPieChart(stats);
    
    // 更新行为列表
    updateBehaviorList(stats);
    
    // 更新数据表格
    updateDataTable(data);
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

// 绘制饼图
function drawPieChart(stats) {
    const canvas = document.getElementById('pieCanvas');
    const ctx = canvas.getContext('2d');
    const pieTotal = document.getElementById('pieTotal');
    const chartLegend = document.getElementById('chartLegend');
    
    canvas.width = 180;
    canvas.height = 180;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
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
function updateBehaviorList(stats) {
    const behaviorList = document.getElementById('behaviorList');
    const behaviors = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 'look_side', 'stand_up'];
    
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

// 下载CSV
document.getElementById('downloadBtn').addEventListener('click', () => {
    alert('CSV文件已保存在 cache_csv 目录下');
});
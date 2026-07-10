"""
空气质量数据分析与可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("空气质量数据分析与可视化")
print("=" * 60)


# ==================== 1. 创建模拟数据 ====================
print("\n1. 数据加载")

np.random.seed(42)

dates = pd.date_range('2010-01-01', '2014-12-31', freq='D')

print(f"生成日期范围: {dates[0]} 到 {dates[-1]}")
print(f"共 {len(dates)} 天")

air_quality = pd.DataFrame({
    'year': dates.year,
    'month': dates.month,
    'day': dates.day,
    'pm2.5': np.maximum(0, np.random.normal(80, 50, len(dates))),
    'DEWP': np.random.normal(0, 10, len(dates)),
    'TEMP': np.random.normal(15, 20, len(dates)),
    'PRES': np.random.normal(1015, 10, len(dates)),
    'cbwd': np.random.choice(['NW', 'NE', 'SW', 'SE'], len(dates)),
    'Iws': np.random.exponential(5, len(dates)),
})

seasonal_factor = np.sin(dates.month / 12 * 2 * np.pi) * 30
air_quality['pm2.5'] = air_quality['pm2.5'] - seasonal_factor + 30

missing_idx = np.random.choice(len(air_quality), 200, replace=False)
air_quality.loc[missing_idx, 'pm2.5'] = np.nan

print(f"数据创建完成，共 {len(air_quality)} 条记录")
print(f"数据形状: {air_quality.shape}")
print(air_quality.head())


# ==================== 2. 数据预处理 ====================
print("\n" + "=" * 60)
print("2. 数据预处理")
print("=" * 60)

# 删除缺失值并创建时间索引
air_quality = air_quality.dropna()
air_quality['datetime'] = pd.to_datetime(air_quality[['year', 'month', 'day']])
air_quality.set_index('datetime', inplace=True)

# 添加季节特征
air_quality['season'] = air_quality['month'].map({
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
    12: 'Winter', 1: 'Winter', 2: 'Winter'
})
print(f"预处理完成，剩余 {len(air_quality)} 条记录")


# ==================== 3. 统计指标 ====================
print("\n" + "=" * 60)
print("3. 统计指标")
print("=" * 60)

print("PM2.5基本统计:")
print(air_quality['pm2.5'].describe())

print("\n各季节PM2.5均值:")
print(air_quality.groupby('season')['pm2.5'].mean())


# ==================== 4. 相关性分析 ====================
print("\n" + "=" * 60)
print("4. 相关性分析")
print("=" * 60)

numeric_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws']
corr_matrix = air_quality[numeric_cols].corr()
print("相关性矩阵:")
print(corr_matrix)
print(f"\n与PM2.5相关性: {corr_matrix['pm2.5'].sort_values(ascending=False)}")


# ==================== 5. 可视化 ====================
print("\n" + "=" * 60)
print("5. 可视化")
print("=" * 60)

fig = plt.figure(figsize=(16, 10))

# 5.1 PM2.5时间序列（2010年）
ax1 = fig.add_subplot(2, 3, 1)
pm25_2010 = air_quality[air_quality['year'] == 2010]['pm2.5']
ax1.plot(pm25_2010.index, pm25_2010.values, linewidth=1, color='steelblue')
ax1.set_title('2010年PM2.5日均值变化', fontsize=11)
ax1.set_xlabel('日期')
ax1.set_ylabel('PM2.5 (μg/m³)')
ax1.grid(True, alpha=0.3)

# 5.2 各季节箱线图（修复：使用tick_labels替代labels）
ax2 = fig.add_subplot(2, 3, 2)
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
season_data = [air_quality[air_quality['season'] == s]['pm2.5'] for s in season_order]

bp = ax2.boxplot(season_data, tick_labels=season_order, patch_artist=True)
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_title('各季节PM2.5分布', fontsize=11)
ax2.set_ylabel('PM2.5 (μg/m³)')
ax2.grid(True, alpha=0.3)

# 5.3 各年份柱状图
ax3 = fig.add_subplot(2, 3, 3)
yearly_mean = air_quality.groupby('year')['pm2.5'].mean()
bars = ax3.bar(yearly_mean.index, yearly_mean.values,
               color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
ax3.set_title('各年份PM2.5年均值', fontsize=11)
ax3.set_xlabel('年份')
ax3.set_ylabel('PM2.5 (μg/m³)')
ax3.grid(True, alpha=0.3)
for bar, val in zip(bars, yearly_mean.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.1f}', ha='center', fontsize=9)

# 5.4 各风向箱线图（修复：使用tick_labels替代labels）
ax4 = fig.add_subplot(2, 3, 4)
wind_order = ['NW', 'NE', 'SW', 'SE']
wind_data = [air_quality[air_quality['cbwd'] == w]['pm2.5'] for w in wind_order]

bp2 = ax4.boxplot(wind_data, tick_labels=wind_order, patch_artist=True)
colors2 = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)
ax4.set_title('各风向PM2.5分布', fontsize=11)
ax4.set_ylabel('PM2.5 (μg/m³)')
ax4.grid(True, alpha=0.3)

# 5.5 相关性热力图
ax5 = fig.add_subplot(2, 3, 5)
im = ax5.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
ax5.set_xticks(range(len(numeric_cols)))
ax5.set_yticks(range(len(numeric_cols)))
ax5.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax5.set_yticklabels(numeric_cols)
ax5.set_title('相关性热力图', fontsize=11)

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center',
                       color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white',
                       fontsize=9)
plt.colorbar(im, ax=ax5, shrink=0.8)

# 5.6 PM2.5 vs 温度散点图
ax6 = fig.add_subplot(2, 3, 6)
sample = air_quality.sample(n=500, random_state=42) if len(air_quality) > 500 else air_quality
ax6.scatter(sample['TEMP'], sample['pm2.5'], alpha=0.4, s=15, c='steelblue')
ax6.set_xlabel('温度 (°C)')
ax6.set_ylabel('PM2.5 (μg/m³)')
ax6.set_title('PM2.5 vs 温度', fontsize=11)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('air_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# ==================== 6. 季节性分析 ====================
print("\n" + "=" * 60)
print("6. 季节性分析")
print("=" * 60)

seasonal_avg = air_quality.groupby('season')['pm2.5'].mean()
print("各季节PM2.5均值:")
print(seasonal_avg)

max_season = seasonal_avg.idxmax()
min_season = seasonal_avg.idxmin()
print(f"\nPM2.5最高季节: {max_season} ({seasonal_avg[max_season]:.1f} μg/m³)")
print(f"PM2.5最低季节: {min_season} ({seasonal_avg[min_season]:.1f} μg/m³)")

print("\n" + "=" * 60)
print("分析完成！图表已保存为 air_quality_analysis.png")
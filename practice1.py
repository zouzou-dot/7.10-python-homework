

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 创建模拟数据 ====================
print("=" * 60)
print("1. 数据加载")
print("=" * 60)


np.random.seed(42)
n = 100

titanic = pd.DataFrame({
    'passenger_id': range(1, n+1),
    'survived': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'name': [f'Passenger_{i}' for i in range(1, n+1)],
    'sex': np.random.choice(['male', 'female'], n),
    'age': np.random.normal(30, 15, n),
    'sibsp': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.3, 0.08, 0.02]),
    'parch': np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
    'fare': np.random.exponential(50, n),
    'embarked': np.random.choice(['C', 'Q', 'S'], n, p=[0.2, 0.1, 0.7]),
    'deck': np.random.choice(['A', 'B', 'C', 'D', np.nan], n, p=[0.1, 0.15, 0.15, 0.1, 0.5])
})

# 添加一些缺失值
titanic.loc[np.random.choice(n, 20, replace=False), 'age'] = np.nan
titanic.loc[np.random.choice(n, 10, replace=False), 'embarked'] = np.nan

print(f"数据集形状: {titanic.shape}")
print(f"列名: {titanic.columns.tolist()}")
print(titanic.head())


# ==================== 2. 缺失值处理 ====================
print("\n" + "=" * 60)
print("2. 缺失值处理")
print("=" * 60)

print("各列缺失值数量:")
print(titanic.isnull().sum())


titanic_clean = titanic.copy()


titanic_clean['age'] = titanic_clean['age'].fillna(titanic_clean['age'].median())
titanic_clean['embarked'] = titanic_clean['embarked'].fillna(titanic_clean['embarked'].mode()[0])
titanic_clean['deck'] = titanic_clean['deck'].fillna('Unknown')

print(f"\n填充后缺失值总数: {titanic_clean.isnull().sum().sum()}")


# ==================== 3. 重复记录处理 ====================
print("\n" + "=" * 60)
print("3. 重复记录处理")
print("=" * 60)

print(f"重复记录数量: {titanic.duplicated().sum()}")


# ==================== 4. 数据类型转换 ====================
print("\n" + "=" * 60)
print("4. 数据类型转换")
print("=" * 60)


titanic_clean['survived'] = titanic_clean['survived'].astype('category')
titanic_clean['pclass'] = titanic_clean['pclass'].astype('category')
titanic_clean['sex'] = titanic_clean['sex'].astype('category')

titanic_clean['family_size'] = titanic_clean['sibsp'] + titanic_clean['parch'] + 1
titanic_clean['age_group'] = pd.cut(titanic_clean['age'],
                                    bins=[0, 12, 18, 35, 60, 100],
                                    labels=['儿童', '青少年', '青年', '中年', '老年'])

print("清洗后数据预览:")
print(titanic_clean[['age', 'age_group', 'sex', 'family_size']].head(10))


# ==================== 5. 验证总结 ====================
print("\n" + "=" * 60)
print("5. 清洗总结")
print("=" * 60)
print(f"✓ 缺失值已处理: {titanic_clean.isnull().sum().sum() == 0}")
print(f"✓ 数据类型已优化")
print(f"✓ 新增特征: family_size, age_group")
print("\n清洗完成！")
# 对于数学建模时间序列分析的一般性步骤

[TOC]


### 1. 导入数据
```python
pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv',parse_dates=['date'],index_col='date')
```

### 2.绘制图像观察序列的特点
```python
plt.plot(df)  # 对于时间序列,直接传入dataframe即可
```

### 3.时间序列的分解
+ 时间序列一般可以分解如下:
  + 加法分解:
    + val = base+trend+seasonal+\theta
  + 乘法分解:
    + val = base*trend*seasonal*\theta
+ 使用`from statsmodel.tsa.seasonal import seasonal_decompose`可以对时间序列分解进行乘法分解和加法分解,代码如下:
```python
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')  #乘法分解
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')
```
`seasonal_decompose()`函数返回元组,其中包含:
+   trend
+ seasonal
+ resid

借助该函数,我们可以绘制去季节化,去趋势化的时间序列图像
1. 去季节化
```python
plt.plot((df.value.values/result_mul.seasonal))
plt.title('时间序列去季节化')
```
2. 去趋势化
```python
detrended = df.value.values - result_mul.trend        # 除以trend会产生所有的年份的销售额期望值均相同的效果
plt.plot(detrended)
plt.title('通过最小二乘拟合来使时间序列去趋势化', fontsize=16)
```

#### 检验时间序列的季节性
使用自相关系数:autocorrelation

### 检验时间序列是否平稳
最常用的检验方式有两种:$ADF$检验,$KPSS$检验
+ $ADF$检验的零假设为序列非平稳,当$p<0.05$时拒绝零假设.也就是说,当我们对序列进行$adf$检验所得结果>0.05时,该序列为非平稳序列
+ $KPSS$检验的零假设为序列平稳,当$p<0.05$时,拒绝零假设.也即是说,当$KPSS$检验所得$p value<0.05$时,序列非平稳
```python
from statsmodels.tsa.stattools import adfuller, kpss
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])

# ADF Test
result = adfuller(df.value.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.value.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
ADF Statistic: 3.14518568930674
p-value: 1.0
Critial Values:
   1%, -3.465620397124192
Critial Values:
   5%, -2.8770397560752436
Critial Values:
   10%, -2.5750324547306476

KPSS Statistic: 1.313675
p-value: 0.010000
Critial Values:
   10%, 0.347
Critial Values:
   5%, 0.463
Critial Values:
   2.5%, 0.574
Critial Values:
   1%, 0.739
```




### 处理时间序列之中的缺失值.
有如下方法:
向后填充；
线性内插；
二次内插；
最邻近平均值；
对应季节的平均值。

### 自相关和偏自相关函数

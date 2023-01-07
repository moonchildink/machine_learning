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
使用自相关系数:autocorrelation:
`from pandas.plotting import autocorrelation_plot`

### 4. 检验时间序列是否平稳
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




### 5. 处理时间序列之中的缺失值.
有如下方法:
向后填充；
线性内插；
二次内插；
最邻近平均值；
对应季节的平均值。

### 6. 自相关和偏自相关函数

#### 自相关函数

+ 自相关函数的定义:
> 自相关（英语：Autocorrelation），也叫序列相关，是一个信号于其自身在不同时间点的互相关。非正式地来说，它就是两次观察之间的相似度对它们之间的时间差的函数。它是找出重复模式（如被噪声掩盖的周期信号），或识别隐含在信号谐波频率中消失的基频的数学工具。它常用于信号处理中，用来分析函数或一系列值，如时域信号。

数学定义:

$r(Z_{t-k},Z_t)=Cov(X_{t-k},X_t)/\sqrt{Var(X_{t-k})Var(X_t)}$

+ 序列的自相关系数以及偏自相关系数图像,使用如下代码生成:

```python
from statsmodel.graphics.tsaplots import plot_acf,plotpacf


# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
plot_acf(df.value.values, lags=50, ax=axes[0])
plot_pacf(df.value.values, lags=50, ax=axes[1])
```


+ ![img.png](自相关系数_偏自相关系数.png)

<img src='自相关系数.png'>





+ 自相关系数的应用:
    + 判断时间序列是否是白噪声:对于白噪声,其自相关系数在不同的`lag`数值下,均为0.实际中,如果样本的自相关系数近似为0,那么我们就可以认为该序列为白噪声序列
    + 对于单调递增的非平稳序列,一般会有如下形式的自相关图象:先是在相当长的一段时间内r>0,而后又一直为负.总体上单调递减,呈三角对称性.
    + 如果自相关系数长期位于0轴的一边,且成正弦波动规律.这是具有周期变化规律的非平稳序列的典型特征
    + 如果自相关系数一直比较小,且始终控制在两倍标准差范围以内.可以推测原序列为随机性非常强的平稳时间序列

#### 偏自相关函数



#### Lag Plot

Lag Plot 也称滞后图.

> A lag plot checks whether a data set or time series is random or not. Random data should not exhibit any identifiable structure in the lag plot. Non-random structure in the lag plot indicates that the underlying data are not random.

[滞后图的解读](https://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm)

<img src='https://www.itl.nist.gov/div898/handbook/eda/gif/lagplot1.gif'>

该图形表示原始时间序列为白噪音,或者说是随即序列.我们无法对该序列做下一步的处理

<img src='https://www.itl.nist.gov/div898/handbook/eda/gif/lagplot2.gif'>

该图形表示原始序列为 弱自相关序列(weak autocorrelation),使用如下方程对其建模:理论上我们可以使用linear regression对其进行回归分析

$Y_i = A_0+A_1*Y_{i-1}+E_i$

<img src='https://www.itl.nist.gov/div898/handbook/eda/gif/lagplot3.gif'>

1. The data come from an underlying autoregressive model with strong positive autocorrelation
2. The data contain no outliers.
3.  Note the tight clustering of points along the diagonal. This is the lag plot signature of a process with strong positive autocorrelation. Such processes are highly non-random--there is strong association between an observation and a succeeding observation.
4. model:

$\begin{equation}{Y_i = A_0+A_1*Y_{i-1}+E_i}\end{equation}$

5. ​	residual standard deviation:

$Y_i = A_0+E_i$

<img src='https://www.itl.nist.gov/div898/handbook/eda/gif/lagplot4.gif'>

1. 可以推测图像来自单周期正弦函数
2. 模型如下: $Y_i = C+\alpha sin(2\pi \omega+\phi)+E_i$
3. 可以忽略异常值

<img src='https://www.statisticshowto.com/wp-content/uploads/2016/11/seasons-lp.png'>

1. 当lag plot呈现正弦特征时,可以推测原始数据有**季节性**特征



### 7. 建模:ARIMA模型

[ARIMA Model - Complete Guide to Time Series Forecasting in Python | ML+ (machinelearningplus.com)](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)

#### 第一步,将模型转换为平稳序列

**差分:difference**






















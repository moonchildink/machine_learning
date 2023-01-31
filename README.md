# 一些笔记



该仓库记录了我学习数学建模过程之中所写的代码和笔记。





#### `sklearn.preprocessing.StandScaler`

该对象用于将随机变量处理为**标准化随机变量**

**定义：**

如果随机变量$X$的数学期望$EX$与方差$DX$均存在，且$DX>0$,就称$X^{\ast} = \dfrac{X-EX}{\sqrt{DX}}$为$X$的标准化随机变量
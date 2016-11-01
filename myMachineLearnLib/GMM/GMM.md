# **GMM算法**

参考：[漫谈 Clustering (3): Gaussian Mixture Model](http://blog.pluskid.org/?p=39)

## 什么是高斯混合模型？
高斯分布相信大家都不陌生，就是那个钟性的曲线。
![wiki高斯分布图](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/360px-Normal_Distribution_PDF.svg.png)

那么何谓高斯分布模型呢？其实就犹如其名字所表达的那样，就是由多个高斯曲线叠加而成的一个模型。
举个例子吧，就上图中的蓝色和绿色的高斯曲线吧。
蓝色和绿色的高斯曲线组合在一起就是一个高斯混合模型了，如果将其看成是分类或者是聚类的话，可以这样理解：
> 其中有两个类别分别为蓝色和绿色，其数据可以想象成下面图片所示，蓝色看成是那个又小又密的点集，而绿色看成是那个类似长条形的数据集。（下面的那种图是从scikit-learn上扣下来的，不要能完全对于高斯曲线，意思对就行了。）
>  
> 如果在图中有一个点，可以分别计算这个点在两个高斯模型下的概率，如果在绿色高斯下的概率大于蓝色的高斯概率，那么这个点就属于绿色阵营了。其基本思想就是这样的了。

![不是太规范的例子](http://scikit-learn.org/stable/_images/sphx_glr_plot_gmm_pdf_0011.png)

**总结：**
高斯混合模型就是由多个高斯曲线组合而成的模型。
如果有一个数据点需要判断其类别，将每一条高斯曲线看成是一个类别，只需要计算此点分别在每个高斯曲线下的概率，将此点划分到概率高的类别中。


## 如何求解高斯混合模型

 数据 => 模型

每个GMM都是由k个Gaussian 分布组成的，每个Gaussian称为一个“Component”，这些“Component”线性加成在一起就组成了GMM的概率密度函数：

$$p(x)=\sum_{k=1}^{k}p(x) \cdot p(x|y) =\sum_{k=1}^{k}\pi_k N(x|u_k ,\Sigma_k)$$

  从中可以看成需要确定这个模型需要确定如下参数：$\pi_k , \mu_k , \Sigma_k$

**ps：发现写漫谈系列的文章真心不错，下面就照搬吧！！**

  现在假设我们有 N 个数据点，并假设它们服从某个分布（记作 p(x) ），现在要确定里面的一些参数的值，例如，在 GMM 中，我们就需要确定 $\pi_k$、$\mu_k$ 和 $\Sigma_k$ 这些参数。 我们的想法是，找到这样一组参数，它所确定的概率分布生成这些给定的数据点的概率最大，而这个概率实际上就等于 $\prod_{i=1}^N p(x_i)$ ，我们把这个乘积称作似然函数 (Likelihood Function)。通常单个点的概率都很小，许多很小的数字相乘起来在计算机里很容易造成浮点数下溢，因此我们通常会对其取对数，把乘积变成加和 $\sum_{i=1}^N \log p(x_i)$，得到 log-likelihood function 。接下来我们只要将这个函数最大化（通常的做法是求导并令导数等于零，然后解方程），亦即找到这样一组参数值，它让似然函数取得最大值，我们就认为这是最合适的参数，这样就完成了参数估计的过程。

GMM的 log-likelihood function 如下：

$$\sum_{i=1}^{n}\log\{\sum_{k=1}^{k} \pi_k N(x_i | \mu_k , \Sigma_k ) \}$$

由于在对数函数里面又有加和，我们没法直接用求导解方程的办法直接求得最大值。为了解决这个问题，我们采取之前从 GMM 中随机选点的办法：分成两步，实际上也就类似于 K-means 的两步。

> 1. **估计数据由每个 Component 生成的概率（并不是每个 Component 被选中的概率）：**
> 对于每个数据 $x_i$ 来说，它由第 k 个 Component 生成的概率为
> $$\gamma(i, k) = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j\mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

> 由于式子里的 $\mu_k$ 和 $\Sigma_k$ 也是需要我们估计的值，我们采用迭代法，在计算$ \gamma(i, k)$ 的时候我们假定 $\mu_k$ 和 $\Sigma_k$ 均已知，我们将取上一次迭代所得的值（或者初始值）。
> 2. **估计每个 Component 的参数： **
> 现在我们假设上一步中得到的 $\gamma(i, k)$ 就是正确的“数据 $x_i $由 Component k 生成的概率”，亦可以当做该 Component 在生成这个数据上所做的贡献，或者说，我们可以看作$ x_i$ 这个值其中有 $\gamma(i, k)x_i $这部分是由 Component k 所生成的。集中考虑所有的数据点，现在实际上可以看作 Component 生成了$ \gamma(1, k)x_1, \ldots, \gamma(N, k)x_N$ 这些点。由于每个 Component 都是一个标准的 Gaussian 分布，可以很容易分布求出最大似然所对应的参数值：
> $$
\begin{aligned}
\mu_k & = \frac{1}{N_k}\sum_{i=1}^N\gamma(i, k)x_i \\
\Sigma_k & = \frac{1}{N_k}\sum_{i=1}^N\gamma(i,
k)(x_i-\mu_k)(x_i-\mu_k)^T
\end{aligned}$$
>其中 $N_k = \sum_{i=1}^N \gamma(i, k)$ ，并且 $\pi_k$ 也顺理成章地可以估计为 $N_k/N$ 。
> 3. **重复迭代前面两步，直到似然函数的值收敛为止。**

##源码：
python版本是我自己实现的，matlab版本是漫谈系列中的。

[matlab版本](http://blog.pluskid.org/?p=39)
[python版本](https://github.com/jie147/myMachineLearningLib/blob/master/com.jie/Unsupervised/GMM.py)

```python
# -*- coding: UTF-8 -*-
# !/usr/bin/python
import random

import numpy as np
from numpy import zeros, array, tile, mean, cov, shape, repeat, sqrt
import numpy.matlib as ml
from numpy.linalg import det, inv


def GMM(X, k, threshold=1e-15):
    n, m = X.shape
    # 挑选个中心点
    centers = array(random.sample(X, k))

    # 初始化 miu ， pi ， sigma
    def init_param():
        # 计算数据点归属那类
        def distance(X, Y):
            n = len(X)
            m = len(Y)
            xx = ml.sum(X * X, axis=1)
            yy = ml.sum(Y * Y, axis=1)
            xy = ml.dot(X, Y.T)
            return tile(xx, (m, 1)).T + tile(yy, (n, 1)) - 2 * xy

        dist = distance(X, centers)
        label = dist.argmin(axis=1)
        # 求解初始 pi 和 sigma
        init_pi = zeros((1, k), dtype=np.float64)
        init_sigma = zeros((k, m, m), dtype=np.float64)
        for i in range(k):
            one_type = X[label == i]
            init_pi[:, i] = 1. * shape(one_type)[0] / n
            init_sigma[i, :, :] = ml.cov(one_type.T)
        return init_pi, init_sigma

    mui = centers
    pi, sigma = init_param()

    print "the shape of init param: mui,pi,sigma"
    print mui.shape, "  ", pi.shape, "  ", sigma.shape
    print "--" * 10
    print mui
    print "--" * 10
    print pi
    print "--" * 10
    print sigma
    print "--" * 10

    # 计算每个点在每个模型中的概率
    def calc_pro(tmp_mui, tmp_sigma):
        p_x = zeros((n, k))
        for i in range(k):
            px_u = X - tmp_mui[i, :]
            # print px_u
            inv_sigma = inv(tmp_sigma[i, :, :])
            # print inv_sigma
            tmp = ml.sum(px_u.dot(inv_sigma) * px_u, axis=1)
            coef = (2 * 3.1415926 * det(inv_sigma)) ** (-1 / 2)
            p_x[:, i] = coef * (ml.exp(-0.5 * tmp))
        return p_x

    L_pre = float('-Inf')
    loop = 0
    while True:
        loop += 1
        print "the number of compute ", loop
        Px = calc_pro(mui, sigma)

        # print Px
        gamma = Px * pi
        gamma = gamma / ml.sum(gamma, axis=0)

        Nk = ml.sum(gamma, axis=0)
        mui = gamma.T.dot(X)
        mui = ml.diag(1 / Nk).dot(mui)
        pi = Nk / n
        for j in range(k):
            x_u = X - mui[j, :]
            sigma[j, :, :] = x_u.T.dot(ml.diag(gamma[:, j]).dot(x_u)) / Nk[j]

        L = ml.sum(ml.log(Px.dot(pi.T)))
        print "the L is :",L
        print "the diff of L is ", L - L_pre
        if (L - L_pre < threshold):
            break
        # if (loop > 300):
        #     break
        L_pre = L

    return mui, sigma, pi, Px


if __name__ == '__main__':
    test_a = np.random.rand(250)
    test_a.resize((50, 5))
    print test_a
    mui, sigma, pi, Px = GMM(test_a, 2)
    print " the mui is :"
    print mui
    print " the sigma is :"
    print sigma
    print " the pi is :"
    print pi
```
# A Study on Stock Price Prediction and Quantitative Strategy - Based on Deep Learning
# 深層学習に基づく株価予測とクオンツ戦略に関する研究
# 基于深度学习的股票价格预测和量化策略研究

#### INTRODUCTION

  In the project, by using the stock data of A-share market, and using LightGBM model to screen 50 factors, the most important 10 factors are selected. After that, we use the BiLSTM model to select the factor combination and establish the quantitative investment strategy. Finally, we make an empirical and back test on the strategy and find that the strategy is better than the market benchmark index, which shows the practical application value of the BiLSTM model in stock price forecasting and quantitative investment.
  
#### 概要
  本研究ではA株の市場全体の株式データを用いて、LightGBMモデルを使用して50の価格ボリュームファクターをフィルタリングし、最も重要度の高い10つの要因を選び出しました。次に、BiLSTMモデルを用いてファクターを結合し、クオンツ投資戦略を構築しました。最後に、この戦略に対して実証的証明かつバックテストを行いました。そして、構築されたクオンツ戦略は市場指標を上回ることを発見し、株価予測とクオンツ投資におけるBiLSTMモデルの応用価値を示しました。それより、クオンツ投資戦略を構築するために、深層学習の有効性を証明しました。



#### 介绍
  本项目通过使用A股全市场的股票数据，并先使用LightGBM模型进行对50个价量因子的筛选，选出重要程度最高的10个因子。之后再用BiLSTM模型选取进行因子组合，建立量化投资策略，最后对该策略进行实证与回测，发现该策略优于市场基准指数，说明了BiLSTM模型在股票价格预测和量化投资的实际应用价值。



#### 项目流程图

![LightGBM-BiLSTM实验步骤](https://images.gitee.com/uploads/images/2021/1009/160622_1c961091_7659950.png "屏幕截图.png")

  本项目先从因子库中选取通过IR检验的50个因子。之后对因子依次进行去极值、缺失值处理、标准化和中性化的因子清洗步骤。再利用LighGBM模型进行因子选择，根据因子重要性进行排序得到前十的因子作为本横截面挑选出来的因子。紧接着利用BiLSTM对挑选出的十个因子进行组合，建立多因子模型。最后构建量化策略，进行策略回测与绩效分析。

#### 实验数据

1. 股票数据：
   A股市场日线数据集包含5872309行数据，即5872309个样本。A股全市场日线数据集数据集有以下11个特征，分别为股票代码（ts_code）、交易日期（trade_date）、开盘价（open）、最高价（hign）、最低价（low）、收盘价（close）、昨收价（pre_close）、涨跌额（change）、涨跌幅（pct_chg）、成交量（vol）和成交额（amount）。中证全指日线数据集包含5057行数据，即包含5057个样本。如表13所示，中证全指日线数据集有以下7个特征，分别依次为交易日期（trade_date）、开盘价（open）、最高价（hign）、最低价（low）、收盘价（close）、交易量（volume）和昨收价（pre_close）。

2. 因子数据：
  本项目节使用如下方式构建价量因子，构建价量因子的基础要素有两点：首先是基础字段，其次是算子。基础字段包括日频的最高价（high）、最低价（low）、开盘价（open）、收盘价（close）、上一日收盘价（pre_close）、成交量（vol）、涨跌（pct_chg）、换手率（turnover_rate）、交易金额（amount）、总市值（total_mv）和复权因子（adj_fator）。本项目通过gplearn提供的基础算子集和自己定义的特殊算子，得到算子列表。通过将基础字段和算子不同的组合，利用遗传规划和人工数据挖掘的方法得到因子生成的公式。

#### 构建的BiLSTM模型

![BiLSTM的网络结构](https://images.gitee.com/uploads/images/2021/1009/161946_684ae133_7659950.png "屏幕截图.png")

  1.层与层之间使用循环神经网络默认的tanh和linear作为激活函数。并且为了防止过拟合加入Dropout，但是如果Dropout使用过大的丢弃比列会出现欠拟合的现象，因此Dropout的丢弃比列取值为0.01。最终模型的BiLSTM循环层的神经元个数为100，采用一层BiLSTM层和三层全连接层，其中BiLSTM层和第一个全连接层之间设置了一个Dropout。

  2.本项目使用数据的数据量较大，所以选用epochs=400，batch_size=1024。模型的损失函数采用均方误差（Mean Square Error，MSE）。其中优化器采用随机梯度下降(Stochastic Gradient Descent，SGD)。随机梯度下降相对于梯度下降（Gradient Descent，GD）有在信息冗余的情况下更能有效的利用信息，前期迭代效果卓越，适合处理大样本的数据这三个优势。由于本实验训练数据量较大，使用SGD的话每次仅用一个样本来迭代，训练的速度很快，可以大大减少训练所花费的时间。本实验使用keras包中的默认值，即lr=0.01、momentum=0.0、decay=0.0和nesterov=False。



#### 策略及回测结果

![策略回测结果](https://images.gitee.com/uploads/images/2021/1009/161242_b3545b61_7659950.png "屏幕截图.png")

![策略净值图](https://images.gitee.com/uploads/images/2021/1009/161309_295841b4_7659950.png "屏幕截图.png")


  1.本项目量化交易策略采用每隔一个月进行换仓（即调仓周期为28个交易日），每次换仓采取等额持股的方式买入BiLSTM预测出的预期收益率最高的25支股票，卖出原本所持有的股票。本文的回测时间和规则如下：

（1）回测时间：从2012年1月到2020年10月。

（2）回测股票池：全A股，剔除特别处理（Special treatment，ST）股票。

（3）交易手续费：买入时支付给券商交易佣金千分之二，卖出时支付给券商交易佣金千分之二，其中单笔交易佣金不满5元券商按5元收取交易佣金。

（4）买卖规则：当天开盘涨停股票不能买入，跌停股票不能卖出。

  2.LightGBM-BiLSTM策略累计收益率为701.00%，远高于中证全指110.40%；年化收益率为29.18%，远高于中证全指9.70%；夏普率为0.77，高于中证全指0.24。这三项回测指标说明LightGBM-BiLSTM策略确实能够给投资者带来更大的收益。LightGBM-BiLSTM策略年化波动率为33.44%大于中证全指26.01%，最大回撤为51.10%小于中证全指58.49%，这两项回测指标说明LightGBM-BiLSTM策略存在一定的风险，特别是很难抵御系统性风险的冲击。年化换手率为11.35，年化交易成本率为2.29%，说明LightGBM-BiLSTM策略不是高频交易策略，交易成本较小。从收益曲线图可以看出LightGBM-BiLSTM策略在前两年的收益率和中证全指相差不大，并没有特别的优势。但从2015年4月左右开始LightGBM-BiLSTM策略的收益率明显好于中证全指的收益率。总体而言，LightGBM-BiLSTM策略的收益率十分可观，但仍然存在一定的风险。


#### 项目展望

 1.本项目在预测股票价格方面，选取的股票收盘价作为预测目标，虽然这一结果最直观，但Bachelier（1900）提出的随机游走假说认为股票的价格服从随机漫步，是不可预测的。之后行为经济学家证明这一观点不完全正确，说明了单纯预测股票收盘价的难度和可解释性不强。因此可以选择股票波动率预测、股票涨跌判断和股票收益率预测等作为未来的研究的方向。

 2.本项目在预测股票价格方面，对比了LSTM、GRU和BiLSTM这三种循环神经网络模型并且说明了BiLSTM预测效果最好，但对比三个模型规模尚少，未来可以深入研究与CNN、DNN和CNN-LSTM等单一或复合模型之间的对比。

 3.本项目在构建量化投资策略方面使用的因子都是技术面的价量因子，因子的种类单一。未来可以选择财务因子、情绪因子、动量因子等不同种类的因子，这些因子分别显示了公司财务状况、投资者的情绪、资产价格的动量效应等。不同种类的因子使预测股票价格更加准确，从而提高策略的性能。同时未来研究还可以适当的加入择时测策略，在预测大盘上涨时增加仓位，在预测大盘下跌时减少仓位，赚取贝塔的钱。

 4.本项目构建的投资组合存在一定的风险，未来可以利用运筹学的二次规划方法对投资组合进行优化，降低投资组合的风险。

 5.本项目在量化投资策略方法采取的是低频交易的策略，未来可以利用股票的tick数据来研究高频策略和超高频策略，优化选股策略。

#### 数据下载（data download）（データ　ダウンロード）
  
  本项目的因子数据集（.h5）和深度学习训练数据（.pkl）已上传至百度云。
   You can download the data in the following URL.
   
    链接(url)：https://pan.baidu.com/s/1B9G0OuPz0Qi3b8WIhPPGKw 
    提取码(password)：52h2 

# sentiment_classification
比赛任务算是Aspect Category Sentiment Analysis

## 相关资料链接

- [Aspect Based Sentiment Analysis总结（一）——任务和数据](https://zhuanlan.zhihu.com/p/81513782)
- https://github.com/BigHeartC/Al_challenger_2018_sentiment_analysis
- https://github.com/jepyh/Sentiment_Analysis_of_User_Reviews
- [AI Challenger 2018情感分析赛道资料汇总](https://blog.csdn.net/lrt366/article/details/89244735)
- [冠军方案](https://tech.meituan.com/2019/01/25/ai-challenger-2018.html)
- [一个fasttext的baseline](https://github.com/panyang/fastText-for-AI-Challenger-Sentiment-Analysis)
- 



## 需要做的工作

1. 数据的预处理：切词（表情符号处理，不同的切分方式）
2. 计算指标的函数（比较好弄）
3. 可以尝试fasttext baseline的那个库，把相关的流程代码直接用到我们的比赛中



## 可以尝试的模型

1. 基于Seq2Seq模型：这个模型比较简单，主要调的可能是数据预处理和词向量方面的东西（[AI2018 第4名](https://mp.weixin.qq.com/s/J6jPxIToPJsA7aSb7wzIuQ)）
2. [基于Aspect的情感极性分析模型](https://github.com/songyouwei/ABSA-PyTorch)，这个的主要问题是他们代码的任务都是基于aspect（每个句子中出现的词），不知道直接把预定好的aspect套到模型里会如何，值得一试，不过有些工作量。


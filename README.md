simple lda for text classification(4 classes)

一、优点
　　1)它可以衡量文档之间的语义相似性。对于一篇文档，我们求出来的主题分布可以看作是对它的一个抽象表示。对于概率分布，我们可以通过一些距离公式（比如KL距离）来计算出两篇文档的语义距离，从而得到它们之间的相似度。
　　2)它可以解决多义词的问题。回想最开始的例子，“苹果”可能是水果，也可能指苹果公司。通过我们求出来的“词语－主题”概率分布，我们就可以知道“苹果”都属于哪些主题，就可以通过主题的匹配来计算它与其他文字之间的相似度。
　　3)它可以排除文档中噪音的影响。一般来说，文档中的噪音往往处于次要主题中，我们可以把它们忽略掉，只保持文档中最主要的主题。
　　4)它是无监督的，完全自动化的。我们只需要提供训练文档，它就可以自动训练出各种概率，无需任何人工标注过程。
　　5)它是跟语言无关的。任何语言只要能够对它进行分词，就可以进行训练，得到它的主题分布。

　　综上所述，主题模型是一个能够挖掘语言背后隐含信息的利器。

二、缺点
模型质量问题
1.模型质量较差，话题出来的无效词较多且较难清洗干净；
2.话题之间，区别不够显著，效果不佳；
3.话题内，词和词的关联性很低。
4.反映不出场景，笔者最开始希望得到的是一个话题，里面有场景词+用户态度、情绪、事件词，构成一个比较完善的系统，但是比较天真...
5.话题命名是个难点，基本词语如果效果差了，话题画像也很难了。


三、LDA使用心得
1.如果要训练一个主题模型用于预测，数据量要足够大；
2.理论上讲，词汇长度越长，表达的主题越明确，这需要一个优秀的词库；
3.如果想要主题划分的更细或突出专业主题，需要专业的词典；
4.LDA的参数alpha对计算效率和模型结果影响非常大，选择合适的alpha可以提高效率和模型可靠性；
5.主题数的确定没有特别突出的方法，更多需要经验；
6.根据时间轴探测热点话题和话题趋势，主题模型是一个不错的选择；
7.前面提到的正面词汇和负面词汇，如何利用，本文没有找到合适的方法；

四、高效的主题模型如何建立？
1.文本要长，要长。不长要想办法拼凑变长
2.语料要好，多下功夫去掉翔
3.规模要大。两层意思，一是文档数大，二是主题数多
4.算法上，plda+能支持中等规模; lightlda能支持大规模（本宝宝有点小贡献，插播个广告）; warplda应该也可以，不过没开源，实现应该不复杂。
5.应用场景要靠谱。直觉上讲，分类等任务还是要有监督的，不太适合无监督的方法去办。而类似基于内容的推荐应用，这种感觉的东西，LDA是靠谱的。
6.短文本别用。要用也要用twitter lda~~~~



延申
supervised lda  https://arxiv.org/pdf/1003.0783.pdf https://github.com/chbrown/slda
twitter lda  https://www.slideshare.net/akshayubhat/twitter-lda https://segmentfault.com/a/1190000010200075 https://github.com/minghui/Twitter-LDA

参考
https://blog.csdn.net/sinat_26917383/article/details/52233341
https://www.cnblogs.com/pinard/p/6908150.html
https://blog.csdn.net/v_JULY_v/article/details/41209515

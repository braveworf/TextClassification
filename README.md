# TextClassification
NLP的一个文本分类任务

首先需要说明的是，这是北邮王晓茹老师的数据挖掘与数据仓库这门课的文本分类的实验。实验要求如下

**实验一文本数据的分类与分析**
【实验目的】
1.掌握数据预处理的方法，对训练集数据进行预处理；
2.掌握文本建模的方法，对语料库的文档进行建模；
3.掌握分类算法的原理，基于有监督的机器学习方法，训练文本分类器；
4.利用学习的文本分类器，对未知文本进行分类判别；
5.掌握评价分类器性能的评估方法。

【实验类型】
数据挖掘算法的设计与编程实现。

【实验要求】
1.文本类别数：>=10类；
2.训练集文档数：>=500000篇；每类平均50000篇。
3.测试集文档数：>=500000篇；每类平均50000篇。
4.分组完成实验，组员数量<=3,个人实现可以获得实验加分。

【实验内容】
利用分类算法实现对文本的数据挖掘，主要包括：
1.语料库的构建，主要包括利用爬虫收集Web文档等；
2.语料库的数据预处理，包括文档建模，如去噪，分词，建立数据字典， 使用词袋模型或主题模型表达文档等；注：使用主题模型，如LDA可以获得实验加分；
3.选择分类算法（朴素贝叶斯（必做）、SVM/其他等），训练文本分类器， 理解所选的分类算法的建模原理、实现过程和相关参数的含义；
4.对测试集的文本进行分类
5.对测试集的分类结果利用正确率和召回率进行分析评价：计算每类正确 率、召回率，计算总体正确率和召回率，以及F-score。

【实验验收】
1.编写实验报告，实验报告内容必须包括对每个阶段的过程描述，以及实 验结果的截图展示。
2.以现场方式验收实验代码。
3.实验完成时间12月15日.

### 数据获取
本次数据部分来源于中国新闻网(http://www.chinanews.com/scroll-news/news10.html), 爬取了从2011年1月至2019年11月的所有文章，共包含体育,台湾,国内,国际,娱乐,房产,文化,汽车,法治,社会,证券,财经共12个类别，每个类别的新闻条数均在10万条以上。在本实验中选取了部分类别，并在每个类别中选取了特定条数的新闻。使用了最基本的python3的requests库和beautifulsoup库即可完成数据爬取过程。

爬虫主要分为2个步骤，首先爬取文章的url存入txt文件中，然后从txt文件读取URL爬取文章。

爬url的方法为: 特定日期的所有的新闻的目录具有特定的格式，例如2019年12月25日的URL如下：http://www.chinanews.com/scroll-news/2019/1225/news.shtml， 页面内容如下图
![avatar](https://imgconvert.csdnimg.cn/aHR0cDovL2kyLnRpaW1nLmNvbS83MDY3MzIvOTFkYWExNGIwMzVjYzg1Yi5wbmc?x-oss-process=image/format,png)
具体代码如下

```python
def getUrls(url):
    req = requests.get(url).text
    if (req == None):
        return
    bf = BeautifulSoup(req, 'html.parser')
    div_bf = bf.find('div', attrs={'class': 'content_list'})
    div_a = div_bf.find_all('div', attrs={'class': 'dd_bt'})
    urltxt = open(b'F:\data\url.txt', 'a', encoding='UTF-8')
    for div in div_a:
        link = div.find('a').get("href")
        urltxt.write(link+'\n')
    urltxt.close()

years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
days = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
for year in range(9):
    for month in range(12):
        print(years[year] + " " + months[month])
        for day in range(31):
        url = "http://www.chinanews.com/scroll-news/" + years[year] + "/" + months[month] + days[day] + "/news.shtml"
        getUrls(url)
```

上一步，获取到的URL的格式为:http://www.chinanews.com/cj/2019/12-26/9044010.shtml, 域名的一级子目录即表示相应的文章的类别。本文爬取所有的文章内容，并根据URL中包含的类别信息在本地归档。爬取文章内容的代码如下

```python
head = requests.head(url)
req = requests.get(url)
req.encoding = 'GB2312'
bf = BeautifulSoup(req.text, 'html.parser')
div = bf.find('div', attrs={'class': 'content'})
h1 = div.find('h1')
head = re.sub(r'\s+', '', h1.get_text())
out = open(filepath + '.txt', 'w', encoding='GB2312', errors='ignore')
out.write(head+'\n')
timediv = div.find('div', attrs={'class': 'left-t'})
time = timediv.get_text().replace(" ", "")[0:16]
out.write(time)
p = div.find('div', attrs={'class': 'left_zw'}).find_all('p', text=True)
for ptext in p:
    out.write('\n'+ptext.text);
out.close()
```

语料库爬取完成之后，本文的内容算是正式开始。文本分类的处理流程如下:

![Markdown](https://imgconvert.csdnimg.cn/aHR0cDovL2kyLnRpaW1nLmNvbS83MDY3MzIvOWM3OTk4ODVhMWM5MWNjMS5wbmc?x-oss-process=image/format,png)
### 文档的表示与特征提取
本文所使用的文档的表示模型是VSM(vector space model,向量空间模型中)，在这种模型中，每个文档以一个词向量的形式表示。

一篇文章如何用词向量表示呢？举个例子，加入一篇文章的内容是这样的

> 我看博主骨骼惊奇，是百年难得一见的练武奇才

然后呢？分词系统将上面的文章分词分成了下面的形式, 把文章分成了一个个的词，并且给出了每个词的词性。(这里推荐使用中科大的分词系统)
> 我/rr 看/v 博/ag 主/ag 骨骼/n 惊奇/an ，/wd 是/vshi 百年/mq 难/a 得/ude3 一/d 见/v 的/ude1 练武/vi 奇才/n 

本文中对于词没有做过多严格要求，本次数据挖掘实验是需要取名词即可，而所有的停用词均为非名词，所以只取名词的操作同时也去掉了所有的停用词。下文所说的停用词的问题在这里得到了解决。对若干篇文章都进行类似的操作，将每篇文章都变成了一个个的名词。如果将所有的名词构成一个无重复元素的集合(w1, w2, w3 ... wn), 这个集合就是字典。对于每篇文章，根据文章中的词是否在字典中存在分别取值0和1。至此，将文章转化为词向量的工作已经完成

如果将上面词向量直接使用的话有什么问题呢？至少有一下几个问题
1.词汇量巨大，造成维度灾难问题
2.一个词出现一次和出现100次是同样的效果，一般来讲，某个词出现的次数多，应该赋予更高的权重
3.有些无意义的词汇，例如"你","我","是","虽然","嗯"之类无意义的词，还有数字，标点符号等词汇需要去除，这些词被称为停用词

所以，需要从众多的词汇中选取具有关键性的词作为字典，为了解决这个问题，本文使用了TF-IDF来选取"最关键性"的词汇，这个过程也称特征提取。
> 阮一峰的博客: [TF-IDF与余弦相似性的应用（一）：自动提取关键词](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html)
> 简单解释:
> TF: Term Frequency即词频，可表示某个词在文章出现的总次数 或者 $TF = \frac{总次数}{文章总词数}$
> 
> 很显然，如果一个词的出现次数越多，表示这个词在文章中越重要，应该给予更高的权重。很显然，单一的词频指标存在问题: 即使某个词出现的频率很高，但是如果它在绝大多数的文章中都多次出现，那么这个词不能代表某类文章，所以应该降低这类词的权重。
> IDF:Inverse Document Frequency即逆文档频率，就是解决上述问题的。
> $$
> IDF = log(\frac{文档总数量}{包含该词的文档数量+1})
> $$
> 分母加一的原因是防止分母为0的情况出现
> 如果某个词在一篇文章中出现的次数很多，在其他文章中出现的次数少，那么这个词就越能代表这篇文档，应该分配更高的权重。
> 最终的TF-IDF权重为TF值和IDF值得乘积

到这一步，任务就已经完成一大半了。

#### 核心代码实现
**一. 分词**
先看看官方文档的[使用案例](https://pynlpir.readthedocs.io/en/latest/tutorial.html), 还可以看看其他的[中科大分词系统的官方文档API](https://pynlpir.readthedocs.io/en/latest/api.html)

```python
import pynlpir
pynlpir.open()
f = open(document_path, 'r')	# document_path是本地txt文章的路径
p = f.read()
segments = pynlpir.segment(p, pos_english=True)		# 分词函数,返回一个列表
# 对分词结果取名词
seg_only_noun = [element[0] for element in segments if element[1] == 'noun']
document_cleaned = ' '.join(seg_only_noun)
pynlpir.close()
```

需要注意的是，如果直接使用这个库会提示缺少license的错误，解决的办法是将[NLPIR系统](https://github.com/NLPIR-team/NLPIR)中`/License/license for a month/NLPIR-ICTCLAS分词系统授权/NLPIR.user"`复制到安装的pynlpir的Data目录之下。本机的Data目录是 C:\Anaconda3\Lib\site-packages\PyNLPIR-0.6.0-py3.7.egg\pynlpir\Data, 其中C:\Anaconda3是python的路径。

**二.数据组织**
本文使用Bunch存储所需要的内容。Bunch的使用方法非常简单，把所有需要的使用的东西全部塞进Bunch里，需要使用的时候直接使用`.`就可以使用。使用Bunch的原因是因为本次数据挖掘使用的数据量比较庞大，在每次测试过程，如果代码从头开始运行，那你肯定会疯掉的。本文使用了Bunch和pickle，使用Bunch将所有需要的数据全部装进一个数据数据结构里。从下面代码的定义可以看出，本文的bunch存的内容有 预测的分类结果集合，测试集的标签，文件名，文件内容(也就是分词后的内容)。

```python
bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])	# 定义Bunch的结构
```

将所有文章的标签和内容存进bunch之后，使用pickle将bunch序列化到本地储存，需要使用时，再将其加载到内存当中。

pickle只需要掌握dump和load两个方法即可，使用方法如下

```python
import pickle
# 序列化数据到本地
with open(word_bag_filepath, 'wb') as file_obj:		 # word_bag_filepath为本地文件路径
    pickle.dump(bunch, file_obj)	# bunch 是需要持久化到本地的数据，支持所有原生python类型
# 从本地加载数据到内存
with open(word_bag_filepath, 'wb') as file_obj:		 # word_bag_filepath为本地文件路径
    bunch = pickle.load(file_obj)
```

所有清洗的代码如下
```python
def clean(raw_data_input_path, word_bag_filepath):
    '''
    函数说明: 对下载的源新闻文档进行分词处理，并且从分词结果中仅取名词
    :param raw_data_input_path: 源文档根目录，其一级子目录存储各个类别的新闻文章，目录名为类别名
    :param word_bag_filepath: 将清洗的结果bunch持久化到词袋文件
    :return: 
    '''
    # 定义Bunch的结构
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

    seg_data_output_head_path = 'F:\\seg\\'		# 分词结果的本地输出路径
    catagorys = os.listdir(raw_data_input_path)		# 一级子目录名是所有的类别名
    bunch.target_name.extend(catagorys)
    pynlpir.open()
    
    # 遍历子目录
    for catagory in catagorys:
        catagory_path = os.path.join(raw_data_input_path, catagory)
        raw_documents = os.listdir(catagory_path)
        # 遍历目录下所有文档
        for document in raw_documents:
            document_path = os.path.join(catagory_path, document)
            # 对每篇文章进行分词处理
            f = open(document_path, 'r')
            p = f.read()
            segments = pynlpir.segment(p, pos_english=True)
            # 对分词结果取名词
            seg_only_noun = [element[0] for element in segments if element[1] == 'noun']
            document_cleaned = ','.join(seg_only_noun)
            #将分词结果加进bunch，并将标签，文章名也加进去
            bunch.filenames.append(document)
            bunch.label.append(catagory)
            bunch.contents.append(document_cleaned)
    pynlpir.close()
    
    with open(word_bag_filepath, 'wb') as file_obj:
    	pickle.dump(bunch, file_obj)
```
**三.提取TF-IDF特征**

请参考 scikit的[TF-IDF文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)， 这里提供了各个参数的详细意义

上文已经给出了TF-IDF的计算方法，可以直接调用sklearn机器学习库即可。这里同样定义了Bunch结构来存储处理后的数据。

```python
tfidf_space = Bunch(target_name = bunch.target_name,	# 所有的分类名称
                        label = bunch.label,			# 所有文章的标签
                        filenames = bunch.filenames,	# 所有文章的文件名
                        tfidf_weight_matrics = [],		# TF-IDF权重矩阵
                        vocabulary = {})				# 词典，上文解释过其意义
```

对于训练集，从所有的文章的词汇中，选取关键的词汇为词典，并将所有的文章转换词向量的形式(词典在转换过程自动生成，可通过上面我给出的官方文档提供的参数调节词典的维度)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.utils import Bunch

with open(bunch_path, 'rb') as file_obj:	## bunch_path 为上文清洗后存储的bunch路径
    bunch = pickle.load(file_obj)
    
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, min_df=0.001)
'''
这里使用了max_df和min_df两个参数指定文档频率的上下界，df表示document frequency
即 如果一个词在10%的文档中都出现(本文共10各类别)，那么该词语不能很好的代表某一个类，所以应该将这个词去掉。min_df同理，如果某个词的频率出现太低，则也无法代表某一类文档，应当忽略。
'''

# 核心代码 将以词汇表示的文档转换为TF-IDF权重矩阵，即转换词向量的形式
tfidf_space.tfidf_weight_matrics = vectorizer.fit_transform(bunch.contents)

tfidf_space.vocabulary = vectorizer.vocabulary_		# 将此过程的词典保存起来。
```

对于测试集，如果照抄测试集的代码，那么两个词向量的词典就不同，所以应该在转换过程，将词典指定为训练集的词典。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.utils import Bunch

with open(bunch_path, 'rb') as file_obj:	## bunch_path 为上文清洗后存储的bunch路径
    bunch = pickle.load(file_obj)
    
with open(train_tfid_path, 'rb') as f:		# 加载训练集 需要使用字典
    trainbunch = pickle.load(f)
    
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, min_df=0.001, vocabulary=trainbunch.vocabulary)
        tfidf_space.tfidf_weight_matrics = vectorizer.fit_transform(bunch.contents)
```

如上，生成了训练集和测试集的特征空间，先将其持久化到本地，供分类器使用

### 朴素贝叶斯
在朴素贝叶斯分类其中，每个元组都被表示成*n*维属性向量X=(x1, x2, ..., xn)的形式，而且一共有K个类，标签分别为C1, C2, ..., Ck。分类的目的是当给定一个元组X时，模型可以预测其应当归属于哪个类别。

$$
\mathrm{P}\left(\mathrm{C}_{i} | \mathrm{X}\right)=\frac{\mathrm{P}\left(\mathrm{X} | \mathrm{C}_{i}\right) \mathrm{P}\left(\mathrm{C}_{i}\right)}{\mathrm{P}(\mathrm{X})}
$$
当且仅当概率 $\mathrm{P}\left(\mathrm{C}_{\mathrm{i}} | \mathrm{X}\right)$ 在$\mathrm{P}\left(\mathrm{C}_{\mathrm{k}} | \mathrm{X}\right)$中取最大值时，即后验概率最大化原则确定预测所属的类别。又因为P(X)是恒定不变的，所以只需要$\mathrm{P}\left(\mathrm{C}_{i} | \mathrm{X}\right)=\mathrm{P}\left(\mathrm{X} | \mathrm{C}_{i}\right) \mathrm{P}\left(\mathrm{C}_{i}\right)$, 最大化即可。

应用朴素贝叶斯分类器时必须满足条件：所有的属性都是条件独立的。也就是说，在给定条件的情况下，属性之间是没有依赖关系的。即
$$
P\left(\mathbf{X} | C_{i}\right)=\prod_{k=1}^{n} P\left(x_{k} | C_{i}\right)=P\left(x_{1} | C_{i}\right) \times P\left(x_{2} | C_{i}\right) \times \ldots \times P\left(x_{n} | C_{i}\right)
$$
本文使用TF-IDF作为文本特征提取，只需要进行相应的加权，使得加权后的$\mathrm{P}\left(\mathrm{C}_{i} | \mathrm{X}\right)$最大得到预测的分类结果。

朴素贝叶斯具有如下的特点:

- 朴素贝叶斯分类器构建非常简单，决策速度极快，并且当新数据可用时(特别是当新数据是附加信息而不是对以前使用的数据进行修改时)，很容易改变概率。
- 在许多应用领域工作良好。
- 易于大规模的维度(100)和数据大小。
- 容易解释做出决定的原因。
- 在开始使用更复杂的分类技术之前，应该先应用NB。

本文使用python3的sklearn.naive_bayes的MultinomialNB库实现，选择alpha参数为0.001。

```python
begintime_train = datetime.datetime.now()
# 训练过程
classifier = MultinomialNB(alpha=0.001).fit(trainset.tfidf_weight_matrics, trainset.label)
endtime_train = datetime.datetime.now()
print("训练完毕，训练时长为：" + str((endtime_train - begintime_train).seconds)+ "秒")

begintime_test = datetime.datetime.now()
# 预测过程
predict = classifier.predict(testset.tfidf_weight_matrics)
endtime_test = datetime.datetime.now()
print("预测完毕，预测时长为：" + str((endtime_test - begintime_test).seconds)+ "秒")
```
### 支持向量机
支持向量机的代码如下, 需要注意的是支持向量机需要使用LinearSVC, 不要使用SVC
```python
from sklearn.svm import LinearSVC
'''
@author: qiuht
LinearSVC
基于liblinear库实现
有多种惩罚参数和损失函数可供选择
训练集实例数量大（大于1万）时也可以很好地进行归一化
既支持稠密输入矩阵也支持稀疏输入矩阵
多分类问题采用one-vs-rest方法实现

SVC
基于libsvm库实现
训练时间复杂度为 [公式]
训练集实例数量大（大于1万）时很难进行归一化
多分类问题采用one-vs-rest方法实现
'''
X_train = trainset.label
y_train = trainset.tfidf_weight_matrics
clf = LinearSVC(C=1, tol=1e-5)
begintime_train = datetime.datetime.now()
clf.fit(y_train, X_train)
endtime_train = datetime.datetime.now()
print("训练完毕，训练时长为：" + str((endtime_train - begintime_train).seconds)+ "秒")

begintime_test = datetime.datetime.now()
predicted = clf.predict(testset.tfidf_weight_matrics)
metrics_result(testset.label, predicted)
endtime_test = datetime.datetime.now()
print("预测完毕，预测时长为：" + str((endtime_test - begintime_test).seconds)+ "秒")
```
### 结果展示
常见的评价指标
准确率: 在每一类中，预测正确的数量 占 这一类的总数量的比例
召回率: 预测正确的数量 占 预测成这一类的比例


```python
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(actual, predict))	# 打印结果

# 打印混淆矩阵
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
confusion_matrix = pd.DataFrame(confusion_matrix(testset.label, predicted), columns=catagorys, index=catagorys)
print(confusion_matrix)
```
**分类结果**: 主要指标: 准确率，召回率，F1 score
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226202837613.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JlZGVtcHRpb24xOTk3,size_16,color_FFFFFF,t_70)
**混淆矩阵**: 第一列为实际的标签，第一行为预测的分类结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226202852414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JlZGVtcHRpb24xOTk3,size_16,color_FFFFFF,t_70)
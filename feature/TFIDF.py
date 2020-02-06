import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.utils import Bunch


def TFIDFVectorSpace(bunch_path, tfidf_path, train_tfid_path = None, bunch = None):
    '''
    函数说明: 生成N篇文档的TF-IDF向量空间
    :param bunch_path: 输入的词袋文件的路径
    :param tfidf_path: 输出的TF-IDF特征空间的路径
    :param train_tfid_path: 训练集的TF-IDF特征空间，其内的词典供测试集使用
    :return:
    '''
    # 读bunch文件
    if bunch is None:
        with open(bunch_path, 'rb') as file_obj:
            bunch = pickle.load(file_obj)
    tfidf_space = Bunch(target_name = bunch.target_name,
                        label = bunch.label,
                        filenames = bunch.filenames,
                        tfidf_weight_matrics = [],
                        vocabulary = {})
    if( train_tfid_path is None ):
        '''        训练集的TF-IDF特征空间        '''
        # 生成词频矩阵并统计TF-IDF值 V1
        print("正在生成训练集的TF-IDF特征空间")
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, min_df=0.001)
        tfidf_space.tfidf_weight_matrics = vectorizer.fit_transform(bunch.contents)
        tfidf_space.vocabulary = vectorizer.vocabulary_
        # 生成词频矩阵并统计TF-IDF值 V2
        # vectorizer = CountVectorizer(max_df=0.5)   # CountVectorizer类会将文本中的词语转换为词频矩阵
        # term_frequency_matrics = vectorizer.fit_transform(bunch.contents)
        # word = vectorizer.get_feature_names()
        # print(word)
        # print(term_frequency_matrics)
        # tranformer = TfidfTransformer(sublinear_tf=True)   # TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值
        # tfidf_space.tfidf_weight_matrics = tranformer.fit_transform(term_frequency_matrics)
        # '''
        # 转换结果: ( i, j ) K 表示在第 i 篇文档中，第 j 个词出现的频数是 K
        # '''
        # print("the tfidf weight matrics:")
        # print(tfidf_space.tfidf_weight_matrics)
    else:
        '''     
        测试集的TF-IDF特征空间： 使用训练集的词典      
        '''
        print("正在生成测试集的TF-IDF特征空间")
        with open(train_tfid_path, 'rb') as f:
            trainbunch = pickle.load(f)
        tfidf_space.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, min_df=0.001, vocabulary=trainbunch.vocabulary)
        tfidf_space.tfidf_weight_matrics = vectorizer.fit_transform(bunch.contents)

    with open(tfidf_path, 'wb') as file_obj:
        pickle.dump(tfidf_space, file_obj)


if __name__ == '__main__':
    # 从类别构建新的bunch
    # catagorys = ['体育','台湾','国内','国际','娱乐','房产','文化','汽车','法治','社会','证券','财经']
    # catagorys = ['体育','台湾','国内','娱乐','房产','文化','汽车','法治','证券']
    catagorys = ['体育','台湾','国内','国际','娱乐','房产','文化','汽车','法治','证券']
    trainbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    trainbunch.target_name.extend(catagorys)
    for item in catagorys:
        path = 'F:\\results\\train_word_bag_'+item+'.dat'
        with open(path, 'rb') as f:
            subbunch = pickle.load(f)
        print("已加载"+path)
        # print(subbunch.keys())
        trainbunch.filenames.extend(subbunch.filenames)
        trainbunch.contents.extend(subbunch.contents)
        trainbunch.label.extend(subbunch.label)

    '''     构建训练集的TF-IDF特征空间        '''
    train_bunch_path = "E:\\DataMining\\0.868\\train_word_bag.dat"
    train_tfidf_path = "F:\\results\\train_tfidf_space_V5.dat"
    '''     构造训练集的TF-IDF向量空间    '''
    TFIDFVectorSpace(train_bunch_path, train_tfidf_path, bunch=trainbunch)

    '''     构建测试集的TF-IDF特征空间    '''
    testbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    testbunch.target_name.extend(catagorys)
    for item in catagorys:
        path = "F:\\results\\test_word_bag_" + item + '.dat'
        with open(path, 'rb') as f:
            subbunch = pickle.load(f)
        print("已加载" + path)
        testbunch.label.extend(subbunch.label)
        testbunch.filenames.extend(subbunch.filenames)
        testbunch.contents.extend(subbunch.contents)
    test_bunch_path = "E:\\DataMining\\0.868\\test_word_bag.dat"
    test_tfidf_path = "F:\\results\\test_tfidf_space_V5.dat"
    TFIDFVectorSpace(test_bunch_path, test_tfidf_path, train_tfidf_path, bunch=testbunch)
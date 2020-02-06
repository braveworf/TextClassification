import pickle

import gensim
from gensim import corpora
from gensim import models
from sklearn.utils import Bunch


def TFIDFVectorSpace(bunch_path, tfidf_path, train_tfid_path = None):
    '''
    函数说明: 生成N篇文档的TF-IDF向量空间
    :param bunch_path: 输入的词袋文件的路径
    :param tfidf_path: 输出的TF-IDF特征空间的路径
    :param train_tfid_path: 训练集的TF-IDF特征空间，其内的词典供测试集使用
    :return:
    '''
    # 读bunch文件
    with open(bunch_path, 'rb') as file_obj:
        bunch = pickle.load(file_obj)
    print("正在分解content")
    contents = [
        [word for word in document.split(" ")]
        for document in bunch.contents
    ]

    tfidf_space = Bunch(target_name = bunch.target_name,
                        label = bunch.label,
                        filenames = bunch.filenames,
                        lda = [],
                        dictionary = [])

    if( train_tfid_path is None ):
        print("正在生成训练集的特征空间")
        with open("F:\\results\\traincontents.dat", 'wb') as file_obj:
            pickle.dump(contents, file_obj)
        tfidf_space.dictionary = corpora.Dictionary(contents)
        corpus = [tfidf_space.dictionary.doc2bow(content) for content in contents]
        tfidf_space.lda = models.LdaModel(corpus, num_topics=10, id2word=tfidf_space.dictionary, passes=500)
    else:
        print("正在生成测试集的特征空间")
        with open("F:\\results\\testcontents.dat", 'wb') as file_obj:
            pickle.dump(contents, file_obj)
        with open(train_tfid_path, 'rb') as f:
            trainbunch = pickle.load(f)
        tfidf_space.dictionary = trainbunch.dictionary
        corpus = [tfidf_space.dictionary.doc2bow(content) for content in contents]
        tfidf_space.lda = models.LdaModel(corpus, num_topics=10, id2word=trainbunch.dictionary, passes=500)

    with open(tfidf_path, 'wb') as file_obj:
        pickle.dump(tfidf_space, file_obj)

if __name__ == '__main__':
    train_bunch_path = "E:\\DataMining\\0.876\\train_word_bag.dat"
    train_tfidf_path = "F:\\results\\train_tfidf_space_gensim1.dat"
    TFIDFVectorSpace(train_bunch_path, train_tfidf_path)

    test_bunch_path = "E:\\DataMining\\0.876\\test_word_bag.dat"
    test_tfidf_path = "F:\\results\\test_tfidf_space_gensim1.dat"
    TFIDFVectorSpace(test_bunch_path, test_tfidf_path, train_tfidf_path)


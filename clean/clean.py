import pickle
import gensim
from gensim import corpora
import pynlpir
import os
from sklearn.utils import Bunch

def initbunch(bunch):
    bunch.target_name=[]
    bunch.label=[]
    bunch.filenames=[]
    bunch.contents=[]

def clean(raw_data_input_path, word_bag_filepath):
    '''
    函数说明: 对下载的源新闻文档进行分词处理，并且从分词结果中仅取名词
    :param raw_data_input_path: 源文档根目录
    :param word_bag_filepath: 将清洗的结果bunch持久化到词袋文件
    :return: 
    '''
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    catagorys = os.listdir(raw_data_input_path)
    bunch.target_name.extend(catagorys)
    pynlpir.open()
    head = 'F:\data'
    finished = ['体育', '台湾']
    for catagory in catagorys:
        if( catagory in finished ):
            continue
        initbunch(bunch)
        catagory_path = os.path.join(raw_data_input_path, catagory)
        years = os.listdir(catagory_path)
        for year in years:
            year_path = os.path.join(head,catagory,year)
            months = os.listdir(year_path)
            for month in months:
                print("cleaning " + catagory + " month: " + month)
                month_path = os.path.join(year_path, month)
                raw_documents = os.listdir(month_path)
                for document in raw_documents:
                    document_path = os.path.join(month_path, document)
                    # 对每篇文章进行分词处理
                    f = open(document_path, 'r')
                    p = f.read()
                    try:
                        segments = pynlpir.segment(p, pos_english=True)
                    except UnicodeDecodeError:
                        print(catagory + ' ' + document + ' UnicodeDecodeError')
                    # 对分词结果取名词
                    seg_only_noun = [element[0] for element in segments if element[1] == 'noun']
                    document_cleaned = ' '.join(seg_only_noun)
                    bunch.filenames.append(document)
                    bunch.label.append(catagory)
                    bunch.contents.append(document_cleaned)
        f = open(word_bag_filepath + catagory + '.dat', 'wb')
        pickle.dump(bunch, f)
    pynlpir.close()

if __name__ == '__main__':
    # 清洗模块测试
    # 训练集
    print("正在清洗训练集数据")
    train_data_path = 'F:\\train'
    train_word_bag = 'F:\\results\\train_word_bag_'
    # clean(train_data_path, train_word_bag)

    #测试集
    print("正在清理测试集数据")
    test_data_path = 'F:\\test'
    test_word_bag = 'F:\\results\\test_word_bag_'
    # clean(test_data_path, test_word_bag)

    #LDA Sample
    print("LDA Sample")
    word_bag_filepath = "F:\\result\\train_word_bag.dat"
    # 读取bunch
    with open(word_bag_filepath, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    # print(bunch.target_name)
    # for t in bunch.contents:
    #     print(t)
    word = [doc.split(',') for doc in bunch.contents]
    dictionary = corpora.Dictionary(word)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in word]
    # for t in doc_term_matrix:
    #     print(t)
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=4, id2word=dictionary, passes=100)
    print(ldamodel.print_topics(num_topics=4, num_words=50))
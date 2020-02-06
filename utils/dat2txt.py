import pickle
from scipy import sparse
import numpy

numpy.set_printoptions(threshold=numpy.inf)

def Dat2Txt(word_bag = None,  tfidf_space = None):
    if(tfidf_space is  None):
        with open(word_bag, 'rb') as file_obj:
            bunch = pickle.load(file_obj)
        transform = open(word_bag+'.txt', 'w')
        for filename, lable, content in zip(bunch.filenames, bunch.label, bunch.contents):
            transform.write(filename + '\t' + lable + '\t' + content + '\n')
        transform.close()
    else:
        with open(tfidf_space, 'rb') as file_obj:
            bunch = pickle.load(file_obj)
        tdm_path = tfidf_space + '_tdm.txt'
        dict_path = tfidf_space + '_dict4.txt'
        # 将IF-IDF权重矩阵写入到txt文件
        tdm = open(tdm_path, 'w')
        # for item in bunch.tfidf_weight_matrics:
        # print(bunch.tfidf_weight_matrics)
        tdm.write(str(bunch.tfidf_weight_matrics))
            # tdm.write('\n')
        tdm.close()
        # 直接储存矩阵(人不可读)
        # sparse.save_npz(tdm_path, bunch.tfidf_weight_matrics)

        # 将词典写入txt文件
        dict = open(dict_path, 'w')
        dict.write('词典维度:' + str(len(bunch.vocabulary)) + '\n')
        dict.write(str(bunch.vocabulary))
        # for item in bunch.vocabulary:
        #     dict.write(item + ',')
        dict.close()

if __name__ == '__main__':
    word_bag = "F:\\results\\train_word_bag_体育.dat"
    tfidf_space = "F:\\results\\train_tfidf_space_V4.dat"
    # Dat2Txt(word_bag = word_bag)
    Dat2Txt(tfidf_space = tfidf_space)

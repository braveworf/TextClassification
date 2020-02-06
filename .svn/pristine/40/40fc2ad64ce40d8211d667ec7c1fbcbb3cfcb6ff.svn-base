import os
import re

from pynlpir import nlpir
import pynlpir
import pickle
import gensim
from gensim import corpora
import pynlpir
import os

# from sklearn.datasets.base import Bunch
from sklearn.utils import Bunch

# nlpir.Init(nlpir.PACKAGE_DIR.encode('utf-8'), nlpir.UTF8_CODE, None)


# 取汉字
# f = open("output.txt", "r", encoding='UTF-8')
# out = open('noun.txt', 'w', encoding='UTF-8')
# content = f.read()

# out.write(' '.join(result))
# out.close()
# f.close()

def check_dir_exist(dir):
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)


def write(filepath, content):
    with open(filepath, "w", encoding='utf-8') as fp:
        fp.write(content)


def readfile(filepath, encoding='utf-8'):
    # 读取文本
    with open(filepath, "rt", encoding=encoding, errors='ignore') as fp:
        content = fp.read()
    return content


def seg(curpus_path, seg_path):
    pynlpir.open()
    check_dir_exist(seg_path)
    cat_folders = os.listdir(curpus_path)
    for folder in cat_folders:
        from_dir = os.path.join(curpus_path, folder)
        to_dir = os.path.join(seg_path, folder)
        check_dir_exist(to_dir)

        files = os.listdir(from_dir)
        i = 0
        for file in files:
            i += 1
            from_file = os.path.join(from_dir, file)
            to_file = os.path.join(to_dir, file)

            nlpir.FileProcess(from_file.encode('UTF-8'), to_file.encode('UTF-8'), True)
            content = readfile(to_file)
            pat = re.compile(u'\s+([\u4e00-\u9fa5]+)/n')
            result = pat.findall(str(content))
            write(to_file, ' '.join(result))


def seg2(curpus_path, seg_path, seg_test_path):
    pynlpir.open(encoding='gbk')
    check_dir_exist(seg_path)
    check_dir_exist(seg_test_path)
    cat_folders = os.listdir(curpus_path)
    i=0
    for folder in cat_folders[::-1]:
        folds_path = os.path.join(curpus_path, folder)
        folds = os.listdir(folds_path)
        for fold in folds:
            files_path = os.path.join(folds_path, fold)
            files = os.listdir(files_path)
            for file in files:
                i+=1
                from_file = os.path.join(files_path, file)
                if i<55000:
                    to_file = os.path.join(seg_path, str(i)+'.txt')
                elif i<125000:
                    to_file = os.path.join(seg_test_path, str(i-55000)+'.txt')
                else:
                    pynlpir.close()
                    return
                nlpir.FileProcess(from_file.encode('UTF-8'), to_file.encode('UTF-8'), True)
                content = readfile(to_file, encoding='gbk')
                pat = re.compile(u'\s+([\u4e00-\u9fa5]+)/n')
                result = pat.findall(str(content))
                write(to_file, ' '.join(result))

def clean(raw_data_input_path, word_bag_filepath):
    '''
    :param word_bag_filepath: 将清洗的结果bunch持久化到词袋文件
    :return:
    '''
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

    catagorys = os.listdir(raw_data_input_path)
    bunch.target_name.extend(catagorys)
    for catagory in catagorys:
        catagory_path = os.path.join(raw_data_input_path, catagory)
        raw_documents = os.listdir(catagory_path)
        for document in raw_documents:
            document_path = os.path.join(catagory_path, document)
            bunch.filenames.append(document)
            bunch.label.append(catagory)
            bunch.contents.append(readfile(document_path))
    with open(word_bag_filepath, 'wb') as file_obj:
        pickle.dump(bunch, file_obj)

seg2(r"D:\coding\entertainment", r"D:\coding\newsdata_seg\entertainment", r"D:\coding\newsdata_test_seg\entertainment")
# clean(r"D:\coding\newsdata_seg", r"D:\coding\word_bag.dat")

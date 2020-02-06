import pickle

from sklearn import metrics
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

f = open("F:\\results\\svm.log", 'w')

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
    print(classification_report(actual, predict))
    f.write('精度:{0:.3f}\n'.format(metrics.precision_score(actual, predict, average='weighted')))
    f.write('召回:{0:0.3f}\n'.format(metrics.recall_score(actual, predict, average='weighted')))
    f.write('f1-score:{0:.3f}\n'.format(metrics.f1_score(actual, predict, average='weighted')))

if __name__ == '__main__':
    # trainset_path = "E:\\DataMining\\train_tfidf_space(降维后).dat"
    trainset_path = "F:\\results\\train_tfidf_space_gensim.dat"
    # testset_path = "E:\\DataMining\\test_tfidf_space(降维后).dat"
    testset_path = "F:\results\test_tfidf_space_gensim.dat"
    with open(trainset_path, 'rb') as file_obj:
        trainset = pickle.load(file_obj)
    '''     训练过程开始      '''
    print("正在训练")
    classifier = SVC(C=1.0, cache_size=800, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                      gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False).fit(trainset.tfidf_weight_matrics, trainset.label)

    with open(testset_path, 'rb') as file_obj:
        testset = pickle.load(file_obj)
    print("正在预测")
    predicted = classifier.predict(testset.tfidf_weight_matrics)

    # 显示记录总体结果
    metrics_result(testset.label, predicted)

    # 打印混淆矩阵
    catagorys = testset.target_name
    # 显示所有列和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    confusion_matrix = pd.DataFrame(confusion_matrix(testset.label, predicted), columns=catagorys, index=catagorys)
    print(confusion_matrix)
    f.write(str(confusion_matrix) + '\n\n')

    for flabel, file_name, expct_cate in zip(testset.label, testset.filenames, predicted):
        # 打印预测错的文档名
        if flabel != expct_cate:
            f.write(file_name + ": 实际类别:" + flabel + " -->预测类别:" + expct_cate + '\n')
    f.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

def read_data():
    df = pd.read_csv('data/data.csv')
    return df

def create_student_success(data,col1,col2,col3,col4):
    data["weighted_col1"]=pd.cut(x=data[col1], bins=[-1,25,50,75,100], labels=[2,3,4,5])
    data["weighted_col2"]=pd.cut(x=data[col2], bins=[-1,25,50,75,100], labels=[2,3,4,5])
    data["weighted_col3"]=pd.cut(x=data[col3], bins=[-1,25,50,75,100], labels=[2,3,4,5])
    data["weighted_col4"]=pd.cut(x=data[col4], bins=[-1,25,50,75,100], labels=[2,3,4,5])

    data["weighted_col1"]=data["weighted_col1"].astype('int64')
    data["weighted_col2"]=data["weighted_col2"].astype('int64')
    data["weighted_col3"]=data["weighted_col3"].astype('int64')
    data["weighted_col4"]=data["weighted_col4"].astype('int64')

    data["StudentSuccess"]=((data[col1]*data['weighted_col1'])+(data[col2]*data['weighted_col2'])+
                      (data[col3]*data['weighted_col3'])+(data[col4]*data['weighted_col4']))/(data["weighted_col1"]+data["weighted_col2"]+data["weighted_col3"]+data["weighted_col4"])

    del data["weighted_col1"]
    del data["weighted_col2"]
    del data["weighted_col3"]
    del data["weighted_col4"]

def visualize_student_success(col1, col2, i):
    plt.subplot(2,3,i)
    plt.title("Student Success for {}".format(col2))
    sns.barplot(data[col2],y=data[col1], data=data)
    plt.xticks(rotation=45)
    plt.ylim(0,85)


def student_success_plot(col1, col2, col3, col4, col5, col6):
    plt.figure(figsize=(17,10), dpi=100)
    visualize_student_success("StudentSuccess", col1, 1)
    visualize_student_success("StudentSuccess", col2, 2)
    visualize_student_success("StudentSuccess", col3, 3)
    visualize_student_success("StudentSuccess", col4, 4)
    visualize_student_success("StudentSuccess", col5, 5)
    visualize_student_success("StudentSuccess", col6, 6)
    plt.tight_layout()
    plt.savefig('images/studentsuccess.jpg')

def class_plot(col1,col2,col3,col4):
    fig, axarr  = plt.subplots(2,2,figsize=(15,10))
    sns.barplot(x='Class', y=col1, data=data, order=['L','M','H'], ax=axarr[0,0])
    sns.barplot(x='Class', y=col2, data=data, order=['L','M','H'], ax=axarr[0,1])
    sns.barplot(x='Class', y=col3, data=data, order=['L','M','H'], ax=axarr[1,0])
    sns.barplot(x='Class', y=col4, data=data, order=['L','M','H'], ax=axarr[1,1])
    fig.suptitle('Student Performance vs Different Class', fontsize = 20)
    plt.tight_layout()
    plt.savefig('images/class.jpg')

def outlier_plot(column):
    plt.figure(figsize=(16,4), dpi=100)
    for i in range(len(column)):
        plt.subplot(1,5,i+1)
        plt.title("{}".format(column[i]))
        plt.boxplot(data[column[i]], whis=1.5 )
    plt.tight_layout()
    plt.savefig('images/outlier.jpg')  

def linear_regression_plot(column):
    plt.figure(figsize=(18,5), dpi=100)
    for i in range(len(column)-1):
        plt.subplot(1,4,i+1)
        sns.regplot(y=data[column[i]], x=data["StudentSuccess"], data=data)

    plt.tight_layout()
    plt.savefig('images/linear.jpg')

def correlation(column):
    correlation=data[column].corr()
    sns.heatmap(correlation, annot=True, linewidths=0.5)
    plt.tight_layout()
    plt.savefig('images/heatmap.jpg')

if __name__ == '__main__':
    data = read_data()
    data = data.rename(columns={'gender': 'Gender', 'raisedhands': 'RaisedHands'})
    
    create_student_success('RaisedHands', 'VisITedResources', 'AnnouncementsView', 'Discussion')
    
    column_numerical = ['RaisedHands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudentSuccess']
    column_nominal = ['Topic', 'NationalITy', 'StudentAbsenceDays', 'ParentschoolSatisfaction', 'Relation', 'StageID']

    student_success_plot('Topic', 'NationalITy', 'StudentAbsenceDays', 'ParentschoolSatisfaction', 'Relation', 'StageID')

    class_plot('RaisedHands', 'VisITedResources', 'AnnouncementsView', 'Discussion')

    outlier_plot(column_numerical)

    linear_regression_plot(column_numerical)

    correlation(column_numerical)





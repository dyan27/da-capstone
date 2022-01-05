import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from visualization import read_data, create_student_success
import sys
import warnings
warnings.filterwarnings('ignore')

def cross_val(model):
    pred = cross_val_score(model, X, y_1, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def linear_plot_student_success(column):
    plt.figure(figsize=(5,5), dpi=100)
    sns.regplot(y=data[column], x=data["Student Success"], data=data)
    plt.tight_layout()
    plt.savefig('images/linear_{}.jpg'.format(column))

def print_linear_status(column):
    x = data[column]
    y = data["Student Success"]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print('###################################')
    print(column)
    print('###################################')
    print('The slope is {}\nThe intercepts are {}\nThe r-square is {}\nThe p-value is {}\nThe standard error is {}'.format(slope, intercept, r_value**2, p_value, std_err))
    

def linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline = Pipeline([('std_scalar', StandardScaler())])
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    model = LinearRegression(normalize=True)
    model.fit(X_train,y_train)
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    pred = model.predict(X_test)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    pd.DataFrame({'True Values': y_test, 'Predicted Values': pred}).plot.scatter(x='True Values', y='Predicted Values')
    plt.title('Student Success Linear Regression Model Fit')
    plt.savefig('images/linear_dist.jpg')

    pd.DataFrame({'Error Values': (y_test - pred)}).plot.kde()
    plt.savefig('images/linear_error.jpg')

    return coeff_df, y_test, y_train, test_pred, train_pred

def ridge_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline = Pipeline([('std_scalar', StandardScaler())])
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
    model.fit(X_train, y_train)
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    pred = model.predict(X_test)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    pd.DataFrame({'True Values': y_test, 'Predicted Values': pred}).plot.scatter(x='True Values', y='Predicted Values')
    plt.savefig('images/ridge_dist.jpg')

    pd.DataFrame({'Error Values': (y_test - pred)}).plot.kde()
    plt.savefig('images/ridge_error.jpg')

    return coeff_df, y_test, y_train, test_pred, train_pred

def lasso_regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline = Pipeline([('std_scalar', StandardScaler())])
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    model = Lasso(alpha=0.1, precompute=True, positive=True, selection='random',random_state=42)
    model.fit(X_train, y_train)
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    pred = model.predict(X_test)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    pd.DataFrame({'True Values': y_test, 'Predicted Values': pred}).plot.scatter(x='True Values', y='Predicted Values')
    plt.savefig('images/lasso_dist.jpg')

    pd.DataFrame({'Error Values': (y_test - pred)}).plot.kde()
    plt.savefig('images/lasso_error.jpg')

    return coeff_df, y_test, y_train, test_pred, train_pred

def logistic_regression (data, X):
    data.Class=[1 if i=='H' else 0 for i in data.Class]
    y = data.Class
    x=(X-np.min(X))/(np.max(X)-np.min(X))
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
    #fit
    model=LogisticRegression(C = 0.5, max_iter = 500)
    model_fit= model.fit(x_train,y_train)
    # coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    pred = model_fit.predict(x_test)

    print('Classification Report of Logistic Regression')
    print()
    print(classification_report(y_test, pred, target_names=['High Class', 'Non-High Class']))
    print('==========================================================')
    print('Confusion Matrix of Logistic Regression')
    print()
    print(pd.DataFrame(confusion_matrix(y_test,pred)))

    return model_fit, x_test, y_test

def ROC_plot(model, x_test, y_test):
    plt.rcParams['figure.figsize'] = [10,8]
    plt.style.use("bmh")
    plt.title("ROC CURVE", fontsize = 15)
    plt.xlabel("False Positive Rate", fontsize = 15)
    plt.ylabel("True Positive Rate", fontsize = 15)

    prob = model.predict_proba(x_test)
    prob_positive = prob[:,1]
    fpr, tpr, threshold = roc_curve(y_test, prob_positive)
    plt.plot(fpr, tpr, color = 'red', label = 'Logistic Regression')
    plt.plot([0,1],[0,1], linestyle = '--', color = 'black', label = 'Random Classifier')
    plt.gca().legend(loc = 'lower right', frameon = True)
    plt.xlim(0,1)
    plt.savefig('images/ROC_Curve.jpg')

    return prob_positive


def print_model_stats(method, coeff_df, y_test, y_train, test_pred, train_pred):
    print('###################################')
    print(method)
    print('###################################')
    print('Test set evaluation:')
    print_evaluate(y_test, test_pred)
    print('Train set evaluation:')
    print_evaluate(y_train, train_pred)
    print(coeff_df)

# def model_stats_table():
#     result = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test_lr, test_pred_lr) , cross_val(LinearRegression())]], 
#                     columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
#     result_rr = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test_rr, test_pred_rr) , cross_val(Ridge())]], 
#                     columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
#     result_la = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test_la, test_pred_la) , cross_val(Lasso())]], 
#                     columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
#     result = result.append(result_rr, ignore_index=True)
#     result = result.append(result_la, ignore_index=True)
    
#     return result



if __name__ == '__main__':
    data = read_data()
    data = data.rename(columns={'gender': 'Gender','NationalITy': 'Nationality', 'PlaceofBirth': 'Place of Birth', 'StageID': 'Stage ID', 
                        'ParentschoolSatisfaction': 'Parentschool Satisfaction', 'StudentAbsenceDays': 'Student Absence Days', 
                        'raisedhands': 'Raised Hands', 'VisITedResources': 'Visited Resources', 'AnnouncementsView': 'Announcements View'})
    
    create_student_success(data, 'Raised Hands', 'Visited Resources', 'Announcements View', 'Discussion')

    X = data[["Raised Hands","Visited Resources","Announcements View","Discussion"]]
    y_1= data["Student Success"]
    y_2 = data['Class']

    sys.stdout = open('data/output.txt', 'w')

    linear_plot_student_success('Raised Hands')
    print_linear_status('Raised Hands')

    linear_plot_student_success('Visited Resources')
    print_linear_status('Visited Resources')

    linear_plot_student_success('Announcements View')
    print_linear_status('Announcements View')

    linear_plot_student_success('Discussion')
    print_linear_status('Discussion')


    coef_lr, y_test_lr, y_train_lr, test_pred_lr, train_pred_lr = linear_regression(X, y_1)
    print_model_stats('Linear Regression', coef_lr, y_test_lr, y_train_lr, test_pred_lr, train_pred_lr)


    # coef_rr, y_test_rr, y_train_rr, test_pred_rr, train_pred_rr = ridge_regression(X, y_1)
    # print_model_stats('Ridge Regression', coef_rr, y_test_rr, y_train_rr, test_pred_rr, train_pred_rr)

    # coef_la, y_test_la, y_train_la, test_pred_la, train_pred_la = lasso_regression(X, y_1)
    # print_model_stats('Lasso Regression', coef_la, y_test_la, y_train_la, test_pred_la, train_pred_la)

    # result_table = model_stats_table()
    # print(result_table)

    # model_fit, x_test, y_test = logistic_regression (data, X)

    # prob_positive = ROC_plot(model_fit, x_test, y_test)
    # print('The ROC AUC score is {}'.format(roc_auc_score(y_test, prob_positive)))

    sys.stdout.close()



    
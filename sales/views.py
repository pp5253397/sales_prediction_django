from django.shortcuts import render,redirect,HttpResponse,HttpResponseRedirect
from .models import RegistrationDetails,SalesDetails,ProductDetails
from .forms import UserDetails,SalesDetailsForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import pandas as pd
from datetime import datetime, timedelta,date
import matplotlib.pyplot as plt
import numpy as np
from django.db.models import Count

import warnings
warnings.filterwarnings("ignore")
import chart_studio.plotly as py
import plotly.offline as pyoff

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

def RegistrationView(request):
    if request.method == 'POST':
        form = UserDetails(request.POST)
        if form.is_valid:
            form.save()
            return redirect('login')
        else:
            return HttpResponse("Something went wrong")
    return render(request,'registration.html')

def LoginView(request):

        if request.POST:
            try:
                model = RegistrationDetails.objects.get(email = request.POST['email'])
                if model.password == request.POST['password']:
                    request.session['user'] = model.username
                    return redirect('dashboard')
                else:
                    return HttpResponse('Something Went Wrong !!!')
            except:
                return HttpResponse('Username Not Found !!')

        return render(request,'login.html')

        
def DashboardView(request):
    if request.session.has_key('user'):
        user = request.session['user']
        product_count = ProductDetails.objects.all().count()
        qty_total = 0
        try:
            top_sales = SalesDetails.objects.filter(sale_date = date.today())
            
            for i in top_sales:
                qty_total += i.qty
        except:
            qty_total = 0

        
        total_sales = SalesDetails.objects.all()
        total = 0

        df = pd.DataFrame(list(SalesDetails.objects.all().values()))#============================
        print(df)

        #==============================Monthly Sales=====================================
        df.to_csv('sales/test.csv',index=False)
        df_sales = pd.read_csv('sales/test.csv')
        df_sales['sale_date'] = pd.to_datetime(df_sales['sale_date'])
        df_sales['sale_date'] = df_sales['sale_date'].dt.year.astype('str') + '-' + df_sales['sale_date'].dt.month.astype('str') + '-01'
        df_sales['sale_date'] = pd.to_datetime(df_sales['sale_date'])
        df_sales = df_sales.groupby('sale_date').qty.sum().reset_index()
        # print(df_sales)
        x_data =df_sales['sale_date']
        y_data = df_sales['qty']

    
        
        plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                         marker_color='green')],
               output_type='div',include_plotlyjs=False,show_link=False, link_text="")
        #=====================================================================================

        #=================SALES DIFFER========================================================
        df_diff = df_sales.copy()
        df_diff['prev_sales'] = df_diff['qty'].shift(1)
        df_diff = df_diff.dropna()
        df_diff['diff'] = (df_diff['qty'] - df_diff['prev_sales'])
        # print(df_diff.head(5))
        plot_differ = plot([Scatter(x=df_diff['sale_date'], y=df_diff['diff'],
                        mode='lines', name='diff',
                         marker_color='green')],
               output_type='div',include_plotlyjs=False,show_link=False, link_text="")

       
        #=======================================================================================

        #=========================SUPERVISED====================================================

        #create dataframe for transformation from time series to supervised
        df_supervised = df_diff.drop(['prev_sales'],axis = 1)
        for inc in range(1,13):
            field_name = 'lag_' + str(inc)
            df_supervised[field_name] = df_supervised['diff'].shift(inc)
        model = smf.ols(formula= 'diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12',data = df_supervised)
        model_fit = model.fit()
        regression_adj_rsq = model_fit.rsquared_adj
        print(regression_adj_rsq)

        df_model = df_supervised.drop(['qty','sale_date'],axis = 1)
        train_set, test_set = df_model[0:-6].values, df_model[-6:].values
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = scaler.fit(train_set)
        train_set = train_set.reshape(train_set.shape[0],train_set.shape[1])
        train_set_scaled = scaler.transform(test_set)

        test_set = test_set.reshape(test_set.shape[0],test_set.shape[1])
        test_set_scaled = scaler.transform(test_set)

        X_train, y_train = train_set_scaled[:, 1:],train_set_scaled[:,0:1]
        X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])

        X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:,0:1]
        X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

        model = Sequential()
        model.add(LSTM(4, batch_input_shape = (1, X_train.shape[1],X_train.shape[2]),stateful = True))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error',optimizer = 'adam') 
        model.fit(X_train, y_train, epochs=100, batch_size = 1,verbose = 1,shuffle=False)

        y_pred = model.predict(X_test,batch_size=1)
        y_pred = y_pred.reshape(y_pred.shape[0],1,y_pred.shape[1])

        pred_test_set = []
        for index in range(0,len(y_pred)):
            print(np.concatenate([y_pred[index],X_test[index]],axis=1))
            pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))

        pred_test_set = np.array(pred_test_set)
        pred_test_set = pred_test_set.reshape(pred_test_set.shape[0],pred_test_set.shape[2])

        pred_test_set_inverted = scaler.inverse_transform(pred_test_set)
        result_list=[]
        sales_dates = list(df_sales[-7:].sale_date)
        act_sales = list(df_sales[-7:].qty)
        for index in range(0,len(pred_test_set_inverted)):
            result_dict = {}
            result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
            result_dict['sale_date'] = sales_dates[index+1]
            result_list.append(result_dict)
        df_result = pd.DataFrame(result_list)
        df_sales_pred = pd.merge(df_sales, df_result, on='sale_date', how='left')
        pred_plot = plot([Scatter(x=df_sales_pred['sale_date'], y=df_sales_pred['qty'],
                        mode='lines', name='actual',
                         marker_color='green'),
                         Scatter(x=df_sales_pred['sale_date'], y=df_sales_pred['pred_value'],
                        mode='lines', name='predicted',
                         marker_color='red')
                         ],
               output_type='div',include_plotlyjs=False,show_link=False, link_text="")
        #=======================================================================================
        for i in total_sales:
            total += i.qty
        return render(request,'dashboard.html',{'qty_total':qty_total,'pred_plot':pred_plot,'user':user,'diff_plot':plot_differ,'count':product_count,'top':top_sales,'total':total,'plot_div': plot_div})
    else:
        return redirect('login')



def error_404(request, exception):
    data = {}
    return render(request,'404.html', data,status=404)

def error_500(request):
    data = {}
    return render(request,'500.html', data,status=500)


def ProductView(request):
    if request.session.has_key('user'):
        user = request.session['user']
        product_details = ProductDetails.objects.all()
        product_count = ProductDetails.objects.all().count()
        page = request.GET.get('page', 1)

        paginator = Paginator(product_details, 6)
        try:
            users = paginator.page(page)
        except PageNotAnInteger:
            users = paginator.page(1)
        except EmptyPage:
            users = paginator.page(paginator.num_pages)
        if request.POST:
            model = ProductDetails()
            model.product_name = request.POST['product']
            model.save()
            
            return redirect('product')

        return render(request,'product.html',{'users':users,'count':product_count,'user':user})
    else:
        return redirect('login')


def SalesView(request):
    if request.session.has_key('user'):
        user = request.session['user']
        product = ProductDetails.objects.all()
 
        sale_details = SalesDetails.objects.filter(sale_date = date.today())
        page = request.GET.get('page', 1)

        paginator = Paginator(sale_details, 5)
        try:
            users = paginator.page(page)
        except PageNotAnInteger:
            users = paginator.page(1)
        except EmptyPage:
            users = paginator.page(paginator.num_pages)
        if request.POST:
            model = SalesDetails()
            model.product_name = request.POST['product_name']
            model.sale_date = request.POST['sale_date']
            model.qty = request.POST['qty']
            model.save()
            print(model.id)
            return HttpResponseRedirect('http://127.0.0.1:8000/sales/')
        return render(request,'addsales.html',{'product':product,'users':users,'user':user})
    else:
        return redirect('login')

def DeleteSale(request,id):
    if request.session.has_key('user'):
      
    
        models = SalesDetails.objects.get(id=id)
        models.delete()
        return HttpResponseRedirect('http://127.0.0.1:8000/sales/')
    else:
        return redirect('login')

def DeleteProduct(request,id):
    if request.session.has_key('user'):
        
        Product = ProductDetails.objects.get(id = id)
        Product.delete()
        return HttpResponseRedirect('http://127.0.0.1:8000/product/')
    else:
        return redirect('login')

def logout(request):
    if request.session.has_key('user'):
        del request.session['user']
        return redirect('login')
    else:
        return redirect('login')


    
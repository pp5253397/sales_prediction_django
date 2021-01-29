from django.urls import path,re_path
from sales import views
urlpatterns = [
    path('register/',views.RegistrationView,name = 'register'),
    path('',views.LoginView,name = 'login'),
    path('dashboard/',views.DashboardView,name = 'dashboard'),
    path('product/',views.ProductView,name = 'product'),
    path('delete/<int:id>/',views.DeleteProduct,name = 'delete'),
    path('sales/',views.SalesView,name = 'addsales'),
    re_path('sales/?page=[0:]',views.SalesView),
    path('sales/delete/<int:id>/',views.DeleteSale,name = 'deletesales'),
    path('logout/',views.logout,name = 'logout')
]


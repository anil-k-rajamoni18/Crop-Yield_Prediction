from django.conf.urls import url
from django.contrib import admin
from classifier import views 


urlpatterns = [
    #url(r'^sequence/$', views.sequence, name='index'),
    url(r'^text/$', views.get_text, name='home'),
    url(r'^result/$', views.result, name='result'),
    url(r'^dataset-details/$', views.dataset, name='dataset'),
    url(r'^machine-learning-algorithm/$', views.algorithm, name='algorithm'),  
    #url(r'^showimage/$', views.showimage, name='showimage'),
    url(r'^png/$', views.getimage,name='tt'),
    url('seq',views.abc,name='seq'),
]
from django.urls import path
from . import views

urlpatterns = [
    #path('home/', views.predictor),
    #path('',views.predictor),
    path('',views.predictornb),
    #path('',views.predictorsvm),
    #path('',views.predictor2),
]

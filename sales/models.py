from django.db import models
from datetime import date

class ProductDetails(models.Model):
    product_name = models.CharField(max_length=90)

    def __str__(self):
        return self.product_name

class SalesDetails(models.Model):
    product_name = models.CharField(max_length=90)
    sale_date = models.CharField(max_length=50)
    qty = models.IntegerField()
    p_id = models.IntegerField(null=True)
    def __str__(self):
        return self.product_name

class RegistrationDetails(models.Model):
    username = models.CharField(max_length=60)
    email = models.EmailField()
    password = models.CharField(max_length=16)

    def __str__(self):
        return self.username
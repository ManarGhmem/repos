# Generated by Django 3.2 on 2024-04-03 08:44

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Cours',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('titre', models.CharField(max_length=255)),
                ('description', models.CharField(max_length=255)),
                ('iframelink', models.CharField(max_length=255)),
            ],
        ),
    ]
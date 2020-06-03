import os
import uuid

from django.db import models
from django.urls import reverse


def directory_path(instance, filename):
    upload_to = 'pictures'
    ext = filename.split('.')[-1]
    if ext == 'mp3':
        upload_to = 'music'
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    return os.path.join(upload_to, filename)


class Music(models.Model):
    name = models.CharField("歌曲名", max_length=30)
    file = models.FileField("文件", upload_to=directory_path, null=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "音乐"


class MusicList(models.Model):
    TYPE_CHOICES = ((0, '愤怒'), (1, '快乐'), (2, '伤感'), (3, '兴奋'), (4, '安静'))
    name = models.CharField("歌单名", max_length=20)
    cover = models.ImageField("图片", upload_to=directory_path, blank=True)
    music = models.ManyToManyField(Music, related_name="musics", verbose_name='歌曲', blank=True)
    mtype = models.PositiveSmallIntegerField('情绪类型', blank=False, default=1, choices=TYPE_CHOICES)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['id']
        verbose_name_plural = "歌单"

from django.contrib import admin

from .models import Music, MusicList
# Register your models here.


class MusicAdmin(admin.ModelAdmin):
    list_display = ('name',)


class MusicListAdmin(admin.ModelAdmin):
    list_display = ('name', 'mtype')


admin.site.register(Music, MusicAdmin)
admin.site.register(MusicList, MusicListAdmin)
import random
from django.shortcuts import render
from django.views.generic import ListView
from django.views import View
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse

from .models import Music, MusicList


def index_redirect(request):
    return HttpResponseRedirect(reverse('musicplayer:music_player'))


class MusicsList(ListView):
    model = MusicList
    template_name = "musicplayer/music_list.html"
    context_object_name = "music_list"


class PlayerView(View):

    def get(self, request, *args, **kwargs):
        return render(request, 'musicplayer/player.html')

    def post(self, request):
        emotion = ['愤怒', '快乐', '悲伤', '惊讶', '平静']
        if request.POST.get('target') == '1':
            # 根据id 获取歌曲名和url
            music = Music.objects.get(id=int(request.POST["music_id"]))
            return JsonResponse({'name': music.name, 'url': music.file.url})
        elif request.POST.get('target') == '2':
            # 根据表情返回歌单
            emotion_id = int(request.POST.get('emotion'))
            l1 = emotion_id
            l2 = l3 = 4
            data = {}
            musics_1, musics_2, musics_3 = [], [], []
            if 0 <= l1 <= 5:
                try:
                    musiclist1 = MusicList.objects.filter(mtype=l1)
                    musiclist1 = musiclist1[random.randint(0, len(musiclist1) - 1)]
                    for music in musiclist1.music.all():
                        musics_1.append((music.name, music.id))
                    data['name1'] = musiclist1.name + "（{}）".format(emotion[l1])
                    data['list1'] = musics_1
                except ValueError:
                    data['name1'] = '歌单未创建({})'.format(emotion[l1])
                    data['list1'] = []
            if 0 <= l2 <= 5:
                try:
                    musiclist2 = MusicList.objects.filter(mtype=l2)
                    musiclist2 = musiclist2[random.randint(0, len(musiclist2) - 1)]
                    for music in musiclist2.music.all():
                        musics_2.append((music.name, music.id))
                    data['name2'] = musiclist2.name + "（{}）".format(emotion[l2])
                    data['list2'] = musics_2
                except ValueError:
                    data['name2'] = '歌单未创建（{}）'.format(emotion[l2])
                    data['list2'] = []
            if 0 <= l3 <= 5:
                try:
                    musiclist3 = MusicList.objects.filter(mtype=l3)
                    musiclist3 = musiclist3[random.randint(0, len(musiclist3) - 1)]
                    for music in musiclist3.music.all():
                        musics_3.append((music.name, music.id))
                    data['name3'] = musiclist3.name + "（{}）".format(emotion[l3])
                    data['list3'] = musics_3
                except ValueError:
                    data['name3'] = '歌单未创建({})'.format(emotion[l3])
                    data['list3'] = []

            return JsonResponse(data)

# Music recommendation system based on facial expression
基于面部表情的音乐推荐系统, 快刀切草莓君的毕业设计, Graduation Project

## 1 项目概述
项目实现的是一个根据用户面部表情推荐音乐的系统，核心算法是卷积神经网络，使用django框架进行包装。项目主要分为面部表情识别和web平台开发两个部分。分别在`FERNetwork`和`FerMusicplayer`中实现，后者调用了前者的训练出的模型。

### 1.1 面部表情识别 FERnetwork
数据集：Fer2013 下载链接见使用方法 神经网络模型：LeNet，AlexNet

**模块介绍**
- `Utils.py`：数据集预处理；读取csv生成npy；合并privatetest和training；删除2种不好的表情
- `Network.py`：神经网络模块；两种模型定义和训练；预测函数接口；
- `FormatPredict.py`：格式化用户输入；滑动窗口识别人脸位置；裁剪出人脸并转换为符合数据集的灰度图；摄像头表情识别demo

### 1.2 DjangoWeb FERmusicplayer
Django项目，主要包含含`faceemotion`和`musicplayer`两个应用以及静态文件和数据库。

**模块介绍**
- Django框架和其他
    - `media`：存放音乐和图片的媒体目录
    - `static`：存放静态文件的目录，使用前需要解压
    - `templates`：存放 html 模板的目录
    - `db.sqlite3`：数据库文件
    - `manage.py`：项目管理入口程序
    - `FERmusicplayer`：Django项目的设置文件
- 应用1 `faceemotion`
    - 功能：表情上传、识别功能，通过ajax方式响应。
    - 引入了`FERnetwork`中的模块和网络权重`checkpoint`
    - 在`views`中调用`FormatPredict`识别表情
- 应用2 `musicplayer`
    - 实现音乐播放器功能
    - 定义音乐模型，实现增删改查
    - 根据用户的表情，推荐相应歌单
    
## 2 使用方法
1. 项目环境 Python3.6+, 使用 pip 安装 requirement.txt 中的依赖项
2. 下载Fer2013数据集 下载链接[Kaggle](https://www.kaggle.com/deadskull7/fer2013) [Zrawberry.com](http://cloud.zrawberry.com/index.php/s/ngwt5QBiR4FMPEj)
3. 训练神经网络 FERnetwork
	- 运行 Utils.py 对数据集预处理
	- 运行 Network.py 训练神经网络模型（指定模型 LeNet or AlexNet）
	- 运行 FormatPredict.py 调用本地摄像头查看效果

4. 运行 django 项目 FERmusicplayer
	- 解压 css 和 js 文件, 创建 db.sqlite3 文件
	- 将训练好的 checkpoint 文件，从 FerNetwork 目录中复制到 `FERmusicplayer/faceemotion/nnSource` 中
	- 执行 `python manage.py makemigrations` 和 `python manage.py migrate` 生成数据库表格
	- 通过 `python manage.py runserver` 运行 django 项目
  
## 其他说明
尚未添加具体开源协议，在使用或参考本项目时用标注 `@Zaaachary @快刀切草莓君 zrawberry.com`即可。

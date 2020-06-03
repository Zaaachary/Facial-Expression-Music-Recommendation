# Music recommendation system based on facial expression
基于面部表情的音乐推荐系统, 快刀切草莓君的毕业设计, Graduation Project

## 使用方法
1. 配置环境 Python3.6+, 安装requirement.txt中的依赖项
2. 下载Fer2013数据集 下载链接[Kaggle](https://www.kaggle.com/deadskull7/fer2013) [Zrawberry.com](http://cloud.zrawberry.com)
3. 训练神经网络 FERnetwork
  - 运行 Utils.py 对数据集预处理
  - 运行 Network.py 训练神经网络模型（指定模型 LeNet or AlexNet）
  - 运行 FormatPredict.py 调用本地摄像头查看效果
4. 运行 django 项目 FERmusicplayer
  - 解压 css 和 js 文件, 创建 db.sqlite3 文件
  - 将训练好的 checkpoint 文件，从 FerNetwork 目录中复制到 `FERmusicplayer/faceemotion/nnSource` 中
  - 执行 `python manage.py makemigrations` 和 `python manage.py migrate` 生成数据库表格
  - 通过 python manage.py runserver 运行 django 项目
  

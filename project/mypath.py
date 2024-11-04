class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'BioVid':
            # folder that contains class labels
            root_dir = 'D:\\large_dataset\\BioVid'

            # Save preprocess data into output_dir
            output_dir = 'F:\\dataset\\frames\\'
            '''视频预处理成图像帧流，后续网络输入读取这个'''

            return root_dir, output_dir

        elif database == "oral":
            # folder that contains class labels
            # 独热编码文件
            label_path = '/root/autodl-tmp/dataset/label/独热编码.xlsx'

            # Save preprocess data into output_dir
            # 图片文件夹
            output_dir = r'E:\yinda\zhujiaqian\project\pic'

            return label_path, output_dir
        # elif database =="inferemce":

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './model/c3d-pretrained.pth'

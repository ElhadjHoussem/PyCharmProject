import numpy as np # linear algebra
import os
import pandas as pd
import cv2

class Data_augmentation:
    def __init__(self, path, image_name):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        self.path = path
        self.name = image_name
        print(path+image_name)
        self.image = cv2.imread(os.path.join(path, image_name))
        print(self.image)
    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image
    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image
    def image_augment(self, save_path):
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        img = self.image.copy()
        img_flip_h = self.flip(img, vflip=True, hflip=False)
        img_flip_v = self.flip(img, vflip=False, hflip=True)
        img_flip_hv = self.flip(img, vflip=True, hflip=True)
        img_rot_90 = self.rotate(img,90)
        img_rot_45 = self.rotate(img,45)
        img_rot_90_ = self.rotate(img,-90)
        img_rot_45_ = self.rotate(img,-45)
        img_gaussian = self.noisy("gauss",img)


        cv2.imwrite(save_path +'/vflip_'+self.name, img_flip_v)
        cv2.imwrite(save_path+'/hflip_'+self.name, img_flip_h)
        cv2.imwrite(save_path+'/hvflip'+self.name, img_flip_hv)
        cv2.imwrite(save_path+'/rot_90'+self.name, img_rot_90)
        cv2.imwrite(save_path+'/rot_45'+self.name, img_rot_45)
        cv2.imwrite(save_path+'/_rot_90__'+self.name, img_rot_90_)
        cv2.imwrite(save_path+'/_rot_45__'+self.name, img_rot_45_)
        cv2.imwrite(save_path+'/GaussianNoise_'+self.name, img_gaussian)
    def DataSetAugmentation(self,file_dir,output_path):
        dir_name_sub_root=""
        for root, dirs, files in os.walk(file_dir):

            test_or_train=["test","train"]
            for i,t in enumerate(test_or_train):
                if root.find(t)!= -1 :
                    dir_name_sub_root=test_or_train[i]

            if len(dirs) == 0:
                dir_name = dir_name_sub_root + "/"+root[-1]



            for file in files:
                raw_image = Data_augmentation(root,file)
                raw_image.image_augment(output_path+"/"+dir_name+"/")
    def DataSetToCsv(path,output_path):
        os.chdir(path)
        lists = os.listdir(path)
        emotions = []
        pixels = []
        usages=[]
        Usage_list=['Training','PublicTest']
        dir_name_sub_root=""

        test_or_train=["test","train"]
        for root, dirs, files in os.walk(path):

            for i,t in enumerate(test_or_train):
                if root.find(t)!= -1 :
                    dir_name_sub_root=test_or_train[i]
                    usage=Usage_list[i]

            if len(dirs) == 0:
                dir_name = dir_name_sub_root + "/"+root[-1]

            for file in files:
                raw_image = np.array(cv2.imread(os.path.join(root, file))).tostring()
                pixels.append(raw_image)
                emotions.append(root[-1])
                usages.append(usage)



        dictP_n = [(root[-1],raw_image,  usage)]

        data = pd.DataFrame(dictP_n, columns=['emotion','pixels','usage'], index = None)
        data = data.sample(frac=1)
        data.to_csv(output_path+"ferAug.csv", index =None)
    def noisy(self,noise_typ,image):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))         for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)
            noisy = image + image * gauss
            return noisy
    def file_to_lists_of_strings(ressource_path,data_file_name,chunck_size=1000):
        with open(ressource_path+data_file_name, "r") as f:
            raws=[]
            line_index=0
            for line in f:
                if line_index==0:
                    line_index+=1
                else:
                    col=line[:-1].split(',')
                    raws.append([str(col[i]) for i in len(col)])
                    if any([col[i]=='' or col[i]=='NaN' for i in range(len(col))]):
                        print('problem'+str(line_index))
                    line_index+=1

            return raws


if __name__ == "__main__":
    path_in = "../Ressources/data/Fer2013Unfolded"
    path_out = "../Ressources/data/Fer2013Augmented2"
    Data_augmentation.DataSetToCsv(path_out,'../Ressources/data/Fer2013')


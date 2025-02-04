import os

pic_path = "D:\keke\matting\mine-matting_0823\\train_code_original\keke_train_data\input"


def rename():
    piclist = os.listdir(pic_path)
    total_num = len(piclist)

    i = 0
    for pic in piclist:
        if pic.endswith("_maskDL.png"):
            old_path = os.path.join(os.path.abspath(pic_path), pic)
            new_path = os.path.join(os.path.abspath(pic_path), 'mask_' + format(str(i), '0>4') + '.png')

            os.renames(old_path, new_path)
            print(u"rename：" + old_path + u"to：" + new_path)
            # print "path：%s, to path：%s" %(old_path,new_path)
            i = i + 1

    print("Total" + str(total_num) + "images rename to:" "0001.png~" + format(str(i - 1), '0>4') + ".png")


rename()

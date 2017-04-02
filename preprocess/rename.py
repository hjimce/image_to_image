import os
import shutil

def rename(origin_root,target_root,flage):
    origin=os.listdir(origin_root)
    for f in origin:
        oldpath=os.path.join(origin_root,f)
        newpath=os.path.join(target_root,flage+f)
        shutil.copy(oldpath,newpath)
rename('gender/first_batch/female','female','b1')
rename('gender/second_batch/female','female','b2')
rename('gender/third_batch/female','female','b3')

rename('gender/first_batch/male','male','b1')
rename('gender/second_batch/male','male','b2')
rename('gender/third_batch/male','male','b3')




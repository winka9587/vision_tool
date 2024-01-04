# echo "handle .jpg -> _color.png"
# bash script/wild6d_relabel_color.sh /data4/cxx/dataset/Wild6D/test_set/bottle/0001/1/images/ /data4/cxx/dataset/Wild6D_manage/test/ .jpg _color.png

# echo "handle _depth.png and _mask.png"
# bash script/data_relabel_suffix.sh /data4/cxx/dataset/Wild6D/test_set/bottle/0001/1/images/ /data4/cxx/dataset/Wild6D_manage/test/ -depth.png _depth.png
# bash script/data_relabel_suffix.sh /data4/cxx/dataset/Wild6D/test_set/bottle/0001/1/images/ /data4/cxx/dataset/Wild6D_manage/test/ -mask.png _oldmask.png

echo "handle .jpg -> _color.png"
bash script/wild6d_relabel_color.sh /data4/cxx/dataset/Wild6D/test_set/bottle/0034/2021-09-16--18-45-29/images/ /data4/cxx/dataset/Wild6D_manage/test/ .jpg _color.png

echo "handle _depth.png and _mask.png"
bash script/data_relabel_suffix.sh /data4/cxx/dataset/Wild6D/test_set/bottle/0034/2021-09-16--18-45-29/images/ /data4/cxx/dataset/Wild6D_manage/test/ -depth.png _depth.png
bash script/data_relabel_suffix.sh /data4/cxx/dataset/Wild6D/test_set/bottle/0034/2021-09-16--18-45-29/images/ /data4/cxx/dataset/Wild6D_manage/test/ -mask.png _oldmask.png

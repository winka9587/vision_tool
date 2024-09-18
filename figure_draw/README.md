# 绘制user study的均值-方差图像

![](/assets/img/2024-06-12-15-12-40.png)

转换为如下的

![](/assets/img/2024-06-12-22-40-36.png)

# 依赖

绘图代码主要依赖于matplotlib, 如果有其他依赖可参考下表：

~~~
# Name                    Version                   Build  Channel
asttokens                 2.4.1                    pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
blas                      1.0                         mkl
blinker                   1.8.2                    pypi_0    pypi
brotli                    1.0.9                h2bbff1b_8
brotli-bin                1.0.9                h2bbff1b_8
ca-certificates           2024.3.11            haa95532_0
certifi                   2024.7.4                 pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
comm                      0.2.2                    pypi_0    pypi
configargparse            1.7                      pypi_0    pypi
contourpy                 1.0.5            py38h59b6b97_0
cycler                    0.11.0             pyhd3eb1b0_0
dash                      2.17.1                   pypi_0    pypi
dash-core-components      2.0.0                    pypi_0    pypi
dash-html-components      2.0.0                    pypi_0    pypi
dash-table                5.0.0                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
executing                 2.0.1                    pypi_0    pypi
fastjsonschema            2.20.0                   pypi_0    pypi
flask                     3.0.3                    pypi_0    pypi
fonttools                 4.51.0           py38h2bbff1b_0
freetype                  2.12.1               ha860e81_0
icu                       73.1                 h6c2663c_0
idna                      3.7                      pypi_0    pypi
importlib-metadata        8.2.0                    pypi_0    pypi
importlib_resources       6.1.1            py38haa95532_1
intel-openmp              2023.1.0         h59b6b97_46320
ipython                   8.12.3                   pypi_0    pypi
ipywidgets                8.1.3                    pypi_0    pypi
itsdangerous              2.2.0                    pypi_0    pypi
jedi                      0.19.1                   pypi_0    pypi
jinja2                    3.1.4                    pypi_0    pypi
jpeg                      9e                   h2bbff1b_1
jsonschema                4.23.0                   pypi_0    pypi
jsonschema-specifications 2023.12.1                pypi_0    pypi
jupyter-core              5.7.2                    pypi_0    pypi
jupyterlab-widgets        3.0.11                   pypi_0    pypi
kiwisolver                1.4.4            py38hd77b12b_0
krb5                      1.20.1               h5b6d351_0
lcms2                     2.12                 h83e58a3_0
lerc                      3.0                  hd77b12b_0
libbrotlicommon           1.0.9                h2bbff1b_8
libbrotlidec              1.0.9                h2bbff1b_8
libbrotlienc              1.0.9                h2bbff1b_8
libclang                  14.0.6          default_hb5a9fac_1
libclang13                14.0.6          default_h8e68704_1
libdeflate                1.17                 h2bbff1b_1
libffi                    3.4.4                hd77b12b_0
libpng                    1.6.39               h8cc25b3_0
libpq                     12.17                h906ac69_0
libtiff                   4.5.1                hd77b12b_0
libwebp-base              1.3.2                h2bbff1b_0
lz4-c                     1.9.4                h2bbff1b_1
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.7.2            py38haa95532_0
matplotlib-base           3.7.2            py38h4ed8f06_0
matplotlib-inline         0.1.7                    pypi_0    pypi
mkl                       2023.1.0         h6b88ed4_46358
mkl-service               2.4.0            py38h2bbff1b_1
mkl_fft                   1.3.8            py38h2bbff1b_0
mkl_random                1.2.4            py38h59b6b97_0
nbformat                  5.10.4                   pypi_0    pypi
nest-asyncio              1.6.0                    pypi_0    pypi
numpy                     1.24.4                   pypi_0    pypi
numpy-base                1.24.3           py38h8a87ada_1
open3d                    0.18.0                   pypi_0    pypi
opencv-python             4.9.0.80                 pypi_0    pypi
openjpeg                  2.4.0                h4fc8c34_0
openssl                   3.0.13               h2bbff1b_2
packaging                 23.2             py38haa95532_0
parso                     0.8.4                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    10.3.0           py38h2bbff1b_0
pip                       23.3.1           py38haa95532_0
pkgutil-resolve-name      1.3.10                   pypi_0    pypi
platformdirs              4.2.2                    pypi_0    pypi
plotly                    5.23.0                   pypi_0    pypi
ply                       3.11                     py38_0
prompt-toolkit            3.0.47                   pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
pygments                  2.18.0                   pypi_0    pypi
pyparsing                 3.0.9            py38haa95532_0
pyqt                      5.15.10          py38hd77b12b_0
pyqt5-sip                 12.13.0          py38h2bbff1b_0
pyrealsense2              2.54.2.5684              pypi_0    pypi
python                    3.8.18               h1aa4202_0
python-dateutil           2.9.0post0       py38haa95532_2
pywin32                   306                      pypi_0    pypi
qt-main                   5.15.2              h19c9488_10
referencing               0.35.1                   pypi_0    pypi
requests                  2.32.3                   pypi_0    pypi
retrying                  1.3.4                    pypi_0    pypi
rpds-py                   0.20.0                   pypi_0    pypi
scipy                     1.10.1                   pypi_0    pypi
setuptools                68.2.2           py38haa95532_0
sip                       6.7.12           py38hd77b12b_0
six                       1.16.0             pyhd3eb1b0_1
sqlite                    3.41.2               h2bbff1b_0
stack-data                0.6.3                    pypi_0    pypi
tbb                       2021.8.0             h59b6b97_0
tenacity                  9.0.0                    pypi_0    pypi
tomli                     2.0.1            py38haa95532_0
tornado                   6.3.3            py38h2bbff1b_0
traitlets                 5.14.3                   pypi_0    pypi
trimesh                   4.4.4                    pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
unicodedata2              15.1.0           py38h2bbff1b_0
urllib3                   2.2.2                    pypi_0    pypi
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wcwidth                   0.2.13                   pypi_0    pypi
werkzeug                  3.0.3                    pypi_0    pypi
wheel                     0.41.2           py38haa95532_0
widgetsnbextension        4.0.11                   pypi_0    pypi
xz                        5.4.6                h8cc25b3_1
zipp                      3.17.0           py38haa95532_0
zlib                      1.2.13               h8cc25b3_1
zstd                      1.5.5                hd43e919_2
~~~
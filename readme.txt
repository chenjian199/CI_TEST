************环境准备*******************
CUDA12.1
Anaconda3

**************************************
1.安装好上述依赖后运行如下命令创建conda环境：
conda create -n AC_Bench python=3.8.17

2.conda创建成功后，运行如下命令启动并进入创建的conda环境
conda activate AC_Bench

3.在上述所创建的AC_Bench环境中使用pip进行依赖安装：
pip install -r requirments.txt

运行方式为(默认使用pytorch后端)：
性能模式：
python run_infer_mode.py --backend PYTORCH \
--model resnet50 \
--scenario Offline \
--data_dir /path/to/imagenetval \
--batch_size 64 \
--duration 60 \
--sample_per_query 100000 \
--querys 1 

精度模式：
python run_infer_mode.py \
 --backend PYTORCH \
 --model resnet50 \
 --scenario Offline \
 --data_dir /path/to/imagenetval \
 --batch_size 80 \
 --duration 60 \
 --sample_per_query 50002 \
 --querys 1 \
 --accuracy True



**************使用说明******************
*本程序运行需要进入上述所创建的conda环境当中，即AC_Bench环境*

**************适配规范说明******************
*厂商需派生backend文件下的backend_base.py做适配
*在backend下新建文件夹文件夹名为后端全称的大写，需要实现__init__.py（init文件应import后端类）,使得主代码可以从__init__.py中找到后端类
*后端类首字母大写，例如nvidia后端的文件夹名称为NVIDIA，类名称为Nidia
*可以参考现有的NVIDIA后端和PYTORCH后端code
*其它无需适配

************代码说明***************
#主程序入口为run_infer_mode.py 的main函数
1.Workload包含了pytorch的模型路径，适配可通过workload.pytorch_model_path访问
2.scenario目前为offline模式
3.Runnable为主推理函数，Runnable会将batch_size张图片组成的列表通过predict接口传给后端，后端自行在predict当中进行batch打包，数据搬运和推理，需返回推理结果
4.针对resnet50,如果厂商未在模型当中进行softmax，则需要实现backend的postprocess进行后处理（可参考pytorch后端）
6.Runnable不可修改

**************主干代码说明***************


 

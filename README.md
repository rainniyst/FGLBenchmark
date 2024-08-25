## 添加自定义算法：

在/algorithm下创建自定义算法。文件和类的命名规则为（以FedAvg为例）：FedAvg.py文件中需包含FedAvgServer和FedAvgClient，且这两个类分别需要继承Base.py中的BaseServer和BaseClient。类似FLGo，Base.py实际上为FedAvg的实现。

### BaseServer中的方法：

run：运行整个算法

communicate：与客户端传递信息

sample：采样客户端

aggregate：聚合

global_evaluate：使用server预留的测试集评估模型

local_validate：使用client的验证集评估模型

### BaseClient中的方法：

train：本地训练









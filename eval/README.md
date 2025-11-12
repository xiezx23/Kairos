## Introduction
本模块支持两种测试方式：
1. 基于[OpenCompass](https://github.com/open-compass/opencompass)开源框架的评测方式：配置简单，同时可以基于LMDeploy等推理引擎加速测试过程
2. 本地实现的评测方式：自由度较高，但不完善

## 前置条件
1. 全精度模型文件：如从HuggingFace下载的模型文件
2. 量化模型参数(及其配置文件)：通过EDQ模块处理输出的模型参数和配置文件

## Method 1: 基于OpenCompass的评测方式
### 1. 参考资料
- [OpenCompass开源项目](https://github.com/open-compass/opencompass)
    - [README](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md#%EF%B8%8F-%EF%B8%8F%E8%AF%84%E6%B5%8B)：详细介绍了使用方法
- [官方教程](https://opencompass.readthedocs.io/zh-cn/latest/index.html)，重点关注：
    - [安装](https://opencompass.readthedocs.io/zh-cn/latest/get_started/installation.html)
    - [快速开始](https://opencompass.readthedocs.io/zh-cn/latest/get_started/quick_start.html)：介绍了OpenCompass的基本用法
    - [学习配置文件](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/config.html): 我们通过编写配置文件来构建模型测试案例
    - [配置数据集](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/datasets.html)：了解其中数据集配置文件目录结构，每种数据集对应许多种配置文件，该资料介绍了它们的区别和用途。
    - [准备模型](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/models.html)：了解如何通过配置构建模型实例，并设定相关参数（如`generation_config`）
    - [使用 LMDeploy 加速评测](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/evaluation_lmdeploy.html)：使用LMDeploy推理引擎加速计算
    - [常见问题](https://opencompass.readthedocs.io/zh-cn/latest/get_started/faq.html)：**强烈建议阅读！**
    
> NOTE: 以上官方教程详细描述了OpenCompass的基本用法，本文不再复述。建议查阅官方教程相关章节了解其细节原理。本文主要说明基于OpenCompass的评测主要操作流程。

### 2. 安装
> 参考前文的`安装`部分官方教程有更详细指引。

为避免与本项目的原依赖库的产生版本冲突，建议创建新的虚拟环境用于评测。例如：

- 面向量化模型的评测环境
    > 由于需要用到`edq_cuda_accel`，出现便利性，可以直接在原有虚拟环境上安装opencompass。
    
    > 在原虚拟环境下运行会出现依赖库的版本冲突，需要手动解决，后文会介绍临时解决方法。

    ```bash
    conda activate edq  # 激活原虚拟环境, 尽量确保edq_cuda_accel可用
    # 支持绝大多数数据集及模型
    pip install -U opencompass

    # 完整安装（支持更多数据集）
    # pip install "opencompass[full]"
    ```    

- 面向全精度模型的评测环境
    > 需要用到LMDeploy推理引擎，其依赖库版本与原虚拟环境的不兼容（torch和cuda等版本要求不一致）
    ```bash
    conda create --name opencompass python=3.10 -y
    conda activate opencompass
    pip install "opencompass[lmdeploy]"
    ```

另外，部分数据集需要额外安装依赖库，例如`HumanEval`数据集。目前暂不考虑这些数据集评测。若后续用到可参考官方教程进行安装。

### 3. 版本冲突临时解决方法：
- `gradio_client`:

    本项目使用的`gradio_client`版本为`0.2.9`，运行opencompass时可能会提示报错:`ImportError: cannot import name 'handle_file' from 'gradio_client'`。
    考虑直接更新到最新版本，或者使用[相关ISSUE](https://github.com/open-compass/opencompass/issues/2044)中提及的版本(`1.9.0`)。
    例如：
    ```bash
     pip install --upgrade gradio_client
    ```
    > 未测试更新版本后，是否对本项目其他模块产生影响。

### 4. 测试方法
- **量化模型**：在`edq_cuda_accel`可用的情况下，使用其进行推理加速。

- **原生模型**：在`LMDeploy`推理引擎可用的情况下，使用其进行推理加速。


1. 构建测试配置文件
    本项目针对不同的模型（base和instruct）及不同的精度（full和quant）提供了相应的测试配置文件模板：
    - full_base：[eval/open_compass/eval_scripts/template/eval_full_base.py](open_compass/eval_scripts/template/eval_full_base.py)
    - quant_base: [eval/open_compass/eval_scripts/template/eval_quant_base.py](open_compass/eval_scripts/template/eval_quant_base.py)
    - full_instruct：[eval/open_compass/eval_scripts/template/eval_full_instruct.py](open_compass/eval_scripts/template/eval_full_instruct.py)
    - quant_instruct: [eval/open_compass/eval_scripts/template/eval_full_instruct.py](open_compass/eval_scripts/template/eval_quant_instruct.py)
        > 对于全精度版本的配置文件，其存在参数`USE_LMDEPLOY`，用于指定是否使用`LMDeploy`推理引擎，默认值为`True`，注意根据实际情况修改。

        > 即运行命令中使用`--a deploy`时设定为`True`，否则为`False`。

    模板文件中给出了较为详细的注释，详细用法请参考模板文件。
    
    补充说明一点，测试配置文件中主要配置两个核心参数。
    > [官方教程](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/config.html)中也有相关介绍
    - `datasets`：指定评测的**数据集列表**，其中每个元素对应某个数据集的一个子集。这意味着可以同时评测多个数据集的不同子集。
    - `models`：指定评测的**模型列表**，其中每个元素对应某个模型的一个实例。这意味着可以同时评测多个模型实例。

    **实际测试时，根据预期的测试场景构建对应的配置文件，建议将其放置在`eval/open_compass/eval_scripts/`目录下，但并无强制要求。**

2. 环境变量配置
    
    **在运行评测前，需要将当前目录切换到项目根目录`/llm-edq`，并且将其添加到`PYTHONPATH`环境变量中**。
    
    这是因为本模块自定义了量化版本模型类(如`QuantHuggingFacewithChatTemplate`)，其不能自动被添加到OpenCompass的`MODELS`类注册器中。这里参考`MMEngine`[教程](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python:~:text=%E9%82%A3%E4%B9%88%E5%B0%B1%E9%9C%80%E8%A6%81,%27CustomOptim%27)的做法，手动指定并注册该类。
    
    为了能够顺利找到并导入该类（位于当前目录下的[eval/open_compass/models/quant_hf_model.py](open_compass/models/quant_hf_model.py)文件），需要将当前目录`/llm-edq`添加到`PYTHONPATH`。
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

3. 基于上述的测试配置文件，我们可以使用命令一键启动评测：
    > 这里仍以模板文件为例，实际使用时修改为对应的测试配置文件。
    ```bash
    ## 评测量化模型(自动检测是否存在`edq_cuda_accel`，由EDQ模块支持)
    opencompass eval/open_compass/eval_scripts/template/eval_quant_instruct.py --debug

    ## 评测全精度模型
    opencompass eval/open_compass/eval_scripts/template/eval_full_instruct.py --debug -a lmdeploy  # 启用LMDeploy推理引擎, 注意将配置文件中的`USE_LMDEPLOY`参数设为`True`
    # or 
    opencompass eval/open_compass/eval_scripts/template/eval_full_instruct.py --debug  # 不启用LMDeploy推理引擎, 注意将配置文件中的`USE_LMDEPLOY`参数设为`False`
    ```
    [其中`--debug`模式用于检查测试流程是否有问题](https://opencompass.readthedocs.io/zh-cn/latest/get_started/quick_start.html#:~:text=%E7%94%B1%E4%BA%8E%20OpenCompass%20%E9%BB%98%E8%AE%A4%E5%B9%B6%E8%A1%8C%E5%90%AF%E5%8A%A8%E8%AF%84%E4%BC%B0%E8%BF%87%E7%A8%8B%EF%BC%8C%E6%88%91%E4%BB%AC%E5%8F%AF%E4%BB%A5%E5%9C%A8%E7%AC%AC%E4%B8%80%E6%AC%A1%E8%BF%90%E8%A1%8C%E6%97%B6%E4%BB%A5%20%2D%2Ddebug%20%E6%A8%A1%E5%BC%8F%E5%90%AF%E5%8A%A8%E8%AF%84%E4%BC%B0%EF%BC%8C%E5%B9%B6%E6%A3%80%E6%9F%A5%E6%98%AF%E5%90%A6%E5%AD%98%E5%9C%A8%E9%97%AE%E9%A2%98%E3%80%82%E5%8C%85%E6%8B%AC%E5%9C%A8%E5%89%8D%E8%BF%B0%E7%9A%84%E6%89%80%E6%9C%89%E6%96%87%E6%A1%A3%E4%B8%AD%EF%BC%8C%E6%88%91%E4%BB%AC%E9%83%BD%E4%BD%BF%E7%94%A8%E4%BA%86%20%2D%2Ddebug%20%E5%BC%80%E5%85%B3%E3%80%82%E5%9C%A8%20%2D%2Ddebug%20%E6%A8%A1%E5%BC%8F%E4%B8%8B%EF%BC%8C%E4%BB%BB%E5%8A%A1%E5%B0%86%E6%8C%89%E9%A1%BA%E5%BA%8F%E6%89%A7%E8%A1%8C%EF%BC%8C%E5%B9%B6%E5%AE%9E%E6%97%B6%E6%89%93%E5%8D%B0%E8%BE%93%E5%87%BA%E3%80%82)，相关日志会输出到`llm_edq/tmp/xxx.log`文件。确认没问题后，可以去除该参数，加速测试过程。



4. 输出信息

    启动测试后，其会在根目录`llm_edq/`下生成两个目录`outputs`和`tmp`。其中`outputs`目录下存储了评测结果，`tmp`目录下存储了评测过程中生成的临时文件（如`debug`时的日志文章）。

    在目录`outputs`主要输出内容如下：

    ```
        outputs/default/
        ├── 20200220_120000
        ├── 20230220_183030     # 每个实验一个文件夹
        │   ├── configs         # 用于记录的已转储的配置文件。
        │   ├── logs            # 推理和评估阶段的日志文件
        │   │   ├── eval
        │   │   └── infer
        │   ├── predictions   # 每个任务的推理结果
        │   ├── results       # 每个任务的评估结果
        │   └── summary       # 单个实验的汇总评估结果
        ├── ...
        
    ```

    参见官方教程的[可视化评估结果](https://opencompass.readthedocs.io/zh-cn/latest/get_started/quick_start.html#:~:text=%E4%B8%AD%E7%9A%84%E6%9B%B4%E5%A4%9A%E5%8F%82%E6%95%B0-,%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AF%84%E4%BC%B0%E7%BB%93%E6%9E%9C,-%E8%AF%84%E4%BC%B0%E5%AE%8C%E6%88%90%E5%90%8E%EF%BC%8C%E8%AF%84%E4%BC%B0)和[结果展示](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/summarizer.html)了解输出内容的更多相关信息。

### 5. 测试时间对比
- 量化模型（仅W8A8Linear)

    是否使用加速库（edq_cuda_accel）及不同batch size下的评测时间对比。测试场景详细见[eval/open_compass/eval_scripts/template/eval_quant_instruct.py](open_compass/eval_scripts/template/eval_quant_instruct.py)。
    > 评测时间测试范围包含了模型构建和参数读取等过程。


    加速库 | batch size | 评测时间 |
    | :-------: | :-------: | :-------: |
    | 无 | 8 | 529.76s |
    | 无 | 16 | 310.64s |
    | 是 | 8 | 176.43s |
    | 是 | 16 | **112.54s** |

- 全精度模型

    是否使用LMDeploy推理引擎的评测时间对比。测试场景详细见[eval/open_compass/eval_scripts/template/eval_full_instruct.py](open_compass/eval_scripts/template/eval_full_instruct.py)。
    加速库 | batch size | 评测时间 |
    | :-------: | :-------: | :-------: |
    | 无 | 8 | 162.92s |
    | 是 | 8 | **65.43s** |

## Method 2: 本地实现的评测方式

> 后续有必要再补充相关说明

### Reference
- https://github.com/haonan-li/CMMLU/blob/master/src/qwen2.py
- https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py

### 预备
本模块用于测试模型在不同评测数据集上的性能表现，默认已经构建得到量化模型。需要准备三方面内容：
- 模型文件：从`HuggingFace`下载的模型文件，假设存储路径为`MODEL_PATH`。
- 量化模型参数：量化模型的参数文件，假设存储路径为`QUANT_WEIGHT_PATH`。
- 评测数据集：评测数据集的文件路径，假设存储路径为`DATASET_PATH`。

### 评测

#### Peak GPU Memory Usage and Inference Latency

#### Perplexity

#### Accuracy
> 仅支持部分数据集评测，后续会根据需求添加更多数据集评测，但也可能不再继续更新。

> 由于数据集的形成不同，我们需要针对每个数据集的格式编写评测脚本。
<!-- > 理论上，对于同一类型的数据集如题型为选择题的数据集），评测脚本的编写是通用的，但是目前未经测试验证。 -->


## 数据集采样
> 由于部分数据集过大，可能无法在有限的时间和计算资源下完成评测。因此，我们提供了数据集采样脚本，从而能够快速得到模型在该数据集上的性能表现。
- MMLU-Pro数据集采样脚本：[eval/data_sampling/mmlup.py](data_sampling/mmlup.py)
- CMMLU数据集采样脚本：[eval/data_sampling/cmmlu.py](data_sampling/cmmlu.py)

目前的采样方式统一为按 category(subject) 分组采样。

注意指定采样输出目录与原数据集目录不同，否则会覆盖原数据集文件。

使用示例：
```bash
cd edq

# 对MMLU-Pro数据集采样
# - num_samples 为采样比例，小于1时为相应原数量的采样比例, 大于1时为绝对采样数量
# - seed 为随机种子，用于复现采样结果
python -m eval.data_sampling.mmlup --dataset_path /data/share/datasets/MMLU-Pro/data --output_dir /data/share/datasets/MMLU-Pro/sampled_data  --num_samples 0.01 --seed 32
```
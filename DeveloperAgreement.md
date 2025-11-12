# Development Standards
## Project Structure
```
<LLM-EDQ>
    |----<Qwen/Qwen2___5-7B-Instruct>   # 模型文件和tokenizer
    |____<llm-edq>                      # (this repo)
             |---- edq                  # 进行模型量化
             |---- kernels              # CUDA算子
             |---- utils                # 工具和通用函数
             |____ README.md
```


## 开发流程
```
每位开发者创建一个自己名字缩写的branch作为开发分支，每个开发分支完成功能实现后通过提交 PR 申请合并
每个commit应该写明修改的重点信息便于版本回溯，每个 PR 需要较为详细的说明更新的内容。
主分支 main 只接受 PR 的更新。
```


## 命名规范
```
文件名:  全小写＋下划线分隔
    例如: quant_model.py

类名称: 大写驼峰命名法
    例如: class W8A8Linear(torch.nn.Module)

函数名称、变量: 全小写＋下划线分隔
    例如: def quantize_tensor_int4(x, q_type = 'A', group_size = -1)

常量:   大写＋下划线分隔
```

## 换行/注释
```
太长的语句应该按照语义换行分割
例如: 
    q_linear.qweight, q_linear.scales, q_linear.scaled_zeros = \
        quant_weight_awq(linear.weight.data.to(torch.float16), q_linear.group_size)

对于较为复杂或者容易混淆的部分，应当用英文或中文进行详细说明
```

